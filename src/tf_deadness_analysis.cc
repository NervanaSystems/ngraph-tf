/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*******************************************************************************

This file is a copy of
Github repository: https://github.com/tensorflow/tensorflow
Revision: 6619dd5fdcad02f087f5758083e2585bdfef9e78
File: tensorflow/tensorflow/compiler/jit/deadness_analysis.cc

*******************************************************************************/
/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "ngraph_utils.h"
#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tf_deadness_analysis.h"

// ALGORITHM OVERVIEW
//
// We map every output produced by each node in the TensorFlow graph (including
// control dependence) into an instance of the Predicate class.  Instances of
// Predicate denote logical formulas and mapping a node `n` to a predicate
// `pred` implies that `n` is executed whenver `pred` is true.  Then we can
// deduce mismatching liveness in the inputs to node by comparing the predicate
// those inputs are mapped to.
//
// Loops are handled pessimistically -- we map Merge nodes with backedges to
// uninterpreted symbols (the same kind we use to represent Switch and _Recv).
// Predicate equality has to hold over all possible assignments to these
// uninterpreted symbols.
namespace tensorflow {

namespace ngraph_bridge {

// namespace {

bool PredicateSequenceEqual(gtl::ArraySlice<Predicate*> lhs,
                            gtl::ArraySlice<Predicate*> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int64 i = 0; i < lhs.size(); i++) {
    if (*lhs[i] != *rhs[i]) {
      return false;
    }
  }
  return true;
}

int64 HashPredicateSequence(Predicate::Kind kind,
                            gtl::ArraySlice<Predicate*> preds) {
  int64 hash = ::tensorflow::hash<Predicate::Kind>()(kind);
  for (Predicate* pred : preds) {
    hash = Hash64Combine(hash, pred->hash());
  }
  return hash;
}



// Common code to create AndPredicate or OrPredicate instances.
Predicate* PredicateFactory::MakeAndOrImpl(gtl::ArraySlice<Predicate*> operands,
                                           bool is_and) {
  Predicate::Kind pred_kind =
      is_and ? Predicate::Kind::kAnd : Predicate::Kind::kOr;
  PredicateSet simplified_ops_set;
  std::vector<Predicate*> simplified_ops;
  for (Predicate* op : operands) {
    // Simplify A&A => A and  A|A => A.
    if (!simplified_ops_set.insert(op).second) {
      continue;
    }
    if (op->kind() == pred_kind) {
      // "Inline" the operands of an inner And/Or into the parent And/Or.
      gtl::ArraySlice<Predicate*> operands =
          is_and ? dynamic_cast<AndPredicate*>(op)->operands()
                 : dynamic_cast<OrPredicate*>(op)->operands();
      for (Predicate* subop : operands) {
        if (simplified_ops_set.insert(subop).second) {
          simplified_ops.push_back(subop);
        }
      }
    } else {
      simplified_ops.push_back(op);
    }
  }

  if (simplified_ops.size() == 1) {
    return simplified_ops[0];
  }
  // Simplify "A&~A=>False" and "A|~A=>True".
  PredicateSet negated_ops;
  for (Predicate* op : simplified_ops) {
    if (op->kind() == Predicate::Kind::kNot) {
      negated_ops.insert(dynamic_cast<NotPredicate&>(*op).operand());
    }
  }
  for (Predicate* op : simplified_ops) {
    if (negated_ops.count(op)) {
      return is_and ? MakeFalse() : MakeTrue();
    }
  }
  std::stable_sort(
      simplified_ops.begin(), simplified_ops.end(),
      [](Predicate* a, Predicate* b) { return a->hash() < b->hash(); });
  return is_and ? Make<AndPredicate>(std::move(simplified_ops))
                : Make<OrPredicate>(std::move(simplified_ops));
}

TensorId InputEdgeToTensorId(const Edge* e) {
  return TensorId(e->src()->name(), e->src_output());
}
std::vector<Predicate*> DeadnessAnalysisImpl::GetIncomingPreds(
    Node* n, DeadnessAnalysisImpl::EdgeKind edge_kind) {
  std::vector<Predicate*> incoming_preds;
  for (const Edge* in_edge : n->in_edges()) {
    bool should_process =
        edge_kind == EdgeKind::kDataAndControl ||
        (in_edge->IsControlEdge() && edge_kind == EdgeKind::kControlOnly) ||
        (!in_edge->IsControlEdge() && edge_kind == EdgeKind::kDataOnly);
    if (should_process) {
      auto it = predicate_map_.find(InputEdgeToTensorId(in_edge));
      CHECK(it != predicate_map_.end()) << in_edge->DebugString();
      incoming_preds.push_back(it->second);
    }
  }
  return incoming_preds;
}
Status DeadnessAnalysisImpl::HandleSwitch(Node* n) {
  std::vector<Predicate*> input_preds =
      GetIncomingPreds(n, EdgeKind::kDataAndControl);
  const Edge* pred_edge;
  TF_RETURN_IF_ERROR(n->input_edge(1, &pred_edge));
  Predicate* true_switch = predicate_factory_.MakeSymbolPredicate(
      TensorId(pred_edge->src()->name(), pred_edge->src_output()),
      /*must_be_true=*/true);
  Predicate* false_switch = predicate_factory_.MakeNotPredicate(true_switch);
  // Output 0 is alive iff all inputs are alive and the condition is false.
  input_preds.push_back(false_switch);
  SetPred(n, 0, predicate_factory_.MakeAndPredicate(input_preds));
  input_preds.pop_back();
  // Output 1 is alive iff all inputs are alive and the condition is true.
  input_preds.push_back(true_switch);
  SetPred(n, 1, predicate_factory_.MakeAndPredicate(input_preds));
  input_preds.pop_back();
  // Control is alive iff any inputs are alive.
  SetPred(n, Graph::kControlSlot,
          predicate_factory_.MakeAndPredicate(input_preds));
  return Status::OK();
}
Status DeadnessAnalysisImpl::HandleMerge(Node* n) {
  // Merge ignores deadness of its control inputs.  A merge that isn't the
  // target of a backedge has is alive iff any of its data inputs are.  We treat
  // the liveness of a merge that is the target of a backedge symbolically.
  bool has_backedge = std::any_of(
      n->in_edges().begin(), n->in_edges().end(), [](const Edge* e) {
        return !e->IsControlEdge() && e->src()->IsNextIteration();
      });
  Predicate* input_data_pred =
      has_backedge
          ? predicate_factory_.MakeSymbolPredicate(TensorId(n->name(), 0),
                                                   /*must_be_true=*/false)
          : predicate_factory_.MakeOrPredicate(
                GetIncomingPreds(n, EdgeKind::kDataOnly));
  SetPred(n, {0, 1, Graph::kControlSlot}, input_data_pred);
  return Status::OK();
}
Status DeadnessAnalysisImpl::HandleRecv(Node* n) {
  // In addition to being alive or dead based on the inputs, a _Recv can also
  // acquire a dead signal from a _Send.
  std::vector<Predicate*> input_preds =
      GetIncomingPreds(n, EdgeKind::kDataAndControl);
  input_preds.push_back(predicate_factory_.MakeSymbolPredicate(
      TensorId(n->name(), 0), /*must_be_true=*/false));
  SetPred(n, {0, Graph::kControlSlot},
          predicate_factory_.MakeAndPredicate(input_preds));
  return Status::OK();
}
Status DeadnessAnalysisImpl::HandleGeneric(Node* n) {
  // Generally nodes are alive iff all their inputs are alive.
  Predicate* pred = predicate_factory_.MakeAndPredicate(
      GetIncomingPreds(n, EdgeKind::kDataAndControl));
  for (int output_idx = 0; output_idx < n->num_outputs(); output_idx++) {
    SetPred(n, output_idx, pred);
  }
  SetPred(n, Graph::kControlSlot, pred);
  return Status::OK();
}
Status DeadnessAnalysisImpl::Populate() {
  std::vector<Node*> rpo;
  GetReversePostOrder(graph_, &rpo, /*stable_comparator=*/{},
                      /*edge_filter=*/[](const Edge& edge) {
                        return !edge.src()->IsNextIteration();
                      });
  // This an abstract interpretation over the deadness propagation semantics of
  // the graph executor.
  for (Node* n : rpo) {
    if (n->IsSwitch()) {
      TF_RETURN_IF_ERROR(HandleSwitch(n));
    } else if (n->IsMerge()) {
      TF_RETURN_IF_ERROR(HandleMerge(n));
    } else if (n->IsControlTrigger()) {
      SetPred(n, Graph::kControlSlot, predicate_factory_.MakeTrue());
    } else if (n->IsRecv() || n->IsHostRecv()) {
      TF_RETURN_IF_ERROR(HandleRecv(n));
    } else {
      TF_RETURN_IF_ERROR(HandleGeneric(n));
    }
  }
  return Status::OK();
}
bool DeadnessAnalysisImpl::HasInputsWithMismatchingDeadness(const Node& node) {
  CHECK(!node.IsMerge());
  if (vlog_) {
    VLOG(2) << "HasInputsWithMismatchingDeadness(" << node.name() << ")";
  }
  Predicate* pred = nullptr;
  for (const Edge* edge : node.in_edges()) {
    auto it = predicate_map_.find(InputEdgeToTensorId(edge));
    CHECK(it != predicate_map_.end()) << edge->DebugString();
    if (vlog_) {
      VLOG(2) << "  " << InputEdgeToTensorId(edge).ToString() << ": "
              << it->second->ToString();
    }
    // Today we just compare the predicates for equality (with some
    // canonicalization/simplification happening before) but we could be more
    // sophisticated here if need be.
    if (pred != nullptr && *pred != *it->second) {
      if (vlog_) {
        VLOG(2) << "HasInputsWithMismatchingDeadness(" << node.name()
                << ") -> true";
      }
      return true;
    }
    pred = it->second;
  }
  if (vlog_) {
    VLOG(2) << "HasInputsWithMismatchingDeadness(" << node.name()
            << ") -> false";
  }
  return false;
}

// We call this function only when we are sure it is a kAnd type node
// TODO: have a faster, non-asserting verison of this function
Status DeadnessAnalysisImpl::GetNodePredicate(const Node& node,
                                              AndPredicate** pred) {
  *pred = nullptr;
  if (node.IsSource() || node.IsSink() || node.IsControlFlow()) {
    return Status::OK();
  }

  for (const Edge* edge : node.out_edges()) {
    auto it = predicate_map_.find(InputEdgeToTensorId(edge));
    CHECK(it != predicate_map_.end()) << edge->DebugString();

    // This node is not control flow but has different output predicates
    if (*pred != nullptr && **pred != *it->second) {
      return errors::Internal(node.name(), "[", node.type_string(), "]",
                              " is a non control flow op. But its outputs have "
                              "different predicates");
    } else {
      // TODO: assert it->second has kind = and
      *pred = static_cast<AndPredicate*>(it->second);
    }
  }
  return Status::OK();
}


void DeadnessAnalysisImpl::Print() const {
  std::vector<TensorId> tensor_ids;
  for (const auto& kv_pair : predicate_map_) {
    tensor_ids.push_back(kv_pair.first);
  }
  std::sort(tensor_ids.begin(), tensor_ids.end());
  for (TensorId tensor_id : tensor_ids) {
    auto it = predicate_map_.find(tensor_id);
    CHECK(it != predicate_map_.end()) << tensor_id.ToString();
    NGRAPH_VLOG(5) << tensor_id.ToString() << " -> " << it->second->ToString();
  }
}
//}  // namespace
DeadnessAnalysis::~DeadnessAnalysis() {}

/*static*/ Status DeadnessAnalysis::Run(
    const Graph& graph, std::unique_ptr<DeadnessAnalysis>* result) {
  std::unique_ptr<DeadnessAnalysisImpl> analysis(
      new DeadnessAnalysisImpl(&graph));

  TF_RETURN_IF_ERROR(analysis->Populate());
  if (NGRAPH_VLOG_IS_ON(5)) {
    analysis->Print();
  }
  *result = std::move(analysis);
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_TF_DISABLE_DEADNESS_CHECK