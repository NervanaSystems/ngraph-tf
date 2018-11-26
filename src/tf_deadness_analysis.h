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
File: tensorflow/tensorflow/compiler/jit/deadness_analysis.h

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
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/gtl/flatset.h"

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
#ifndef NGRAPH_TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_
#define NGRAPH_TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace ngraph_bridge {

class AndPredicate;
class OrPredicate;
class NotPredicate;
class SymbolPredicate;

// namespace {
// Represents a logical predicate, used as described in the algorithm overview
// above.
class Predicate {
 public:
  enum class Kind { kAnd, kOr, kNot, kSymbol };
  virtual string ToString() const = 0;
  virtual bool operator==(const Predicate& other) const = 0;
  virtual bool operator!=(const Predicate& other) const {
    return !(*this == other);
  }
  int64 hash() const { return hash_; }
  virtual Kind kind() const = 0;
  virtual ~Predicate() {}

 protected:
  explicit Predicate(int64 hash) : hash_(hash) {}

 private:
  const int64 hash_;
};


// Creates and owns Predicate instances.  Simplifies predicates as it creates
// them.
class PredicateFactory {
 public:
  Predicate* MakeAndPredicate(gtl::ArraySlice<Predicate*> operands) {
    return MakeAndOrImpl(operands, /*is_and=*/true);
  }
  Predicate* MakeOrPredicate(gtl::ArraySlice<Predicate*> operands) {
    return MakeAndOrImpl(operands, /*is_and=*/false);
  }
  Predicate* MakeNotPredicate(Predicate* pred) {
    return Make<NotPredicate>(pred);
  }
  Predicate* MakeSymbolPredicate(TensorId tensor_id, bool must_be_true) {
    return Make<SymbolPredicate>(tensor_id, must_be_true);
  }
  Predicate* MakeTrue() { return MakeAndPredicate({}); }
  Predicate* MakeFalse() { return MakeOrPredicate({}); }

 private:
  template <typename PredicateT, typename... Args>
  Predicate* Make(Args... args) {
    std::unique_ptr<PredicateT> pred(
        new PredicateT(std::forward<Args>(args)...));
    predicate_storage_.emplace_back(std::move(pred));
    return predicate_storage_.back().get();
  }
  Predicate* MakeAndOrImpl(gtl::ArraySlice<Predicate*> operands, bool is_and);
  struct PredicatePtrHash {
    size_t operator()(const Predicate* pred) const { return pred->hash(); }
  };
  struct PredicatePtrEq {
    size_t operator()(const Predicate* a, const Predicate* b) const {
      return *a == *b;
    }
  };
  using PredicateSet =
      gtl::FlatSet<Predicate*, PredicatePtrHash, PredicatePtrEq>;
  std::vector<std::unique_ptr<Predicate>> predicate_storage_;
};

// This analyzes a TensorFlow graph to identify nodes which may have partially
// dead inputs (i.e. these nodes may have some dead inputs and some alive
// inputs).
//
// For example, the ADD node in the following graph
//
//      V0  PRED0    V1  PRED1
//       |    |       |    |
//       v    v       v    v
//       SWITCH       SWITCH
//          |            |
//          +---+   + ---+
//              |   |
//              v   v
//               ADD
//
// can have its inputs independently dead or alive based on the runtime values
// of PRED0 and PRED1.
//
// It is tempting to call this a liveness analysis but I avoided that because
// "liveness" already has other connotations.
class DeadnessAnalysis {
 public:
  // Returns true if `node` may have some live inputs and some dead inputs.
  //
  // This is a conservatively correct routine -- if it returns false then `node`
  // is guaranteed to not have inputs with mismatching liveness, but not the
  // converse.
  //
  // REQUIRES: node is not a Merge operation.
  virtual bool HasInputsWithMismatchingDeadness(const Node& node) = 0;

  // Prints out the internal state of this instance.  For debugging purposes
  // only.
  virtual void Print() const = 0;
  virtual ~DeadnessAnalysis();
  // Run the deadness analysis over `graph` and returns an error or a populated
  // instance of DeadnessAnalysis in `result`.
  static Status Run(const Graph& graph,
                    std::unique_ptr<DeadnessAnalysis>* result);

  // This returns an AndPredicate, otherwise it returns nullptr
  virtual Status GetNodePredicate(const Node& node, AndPredicate** pred) = 0;

  virtual Predicate* CreateTestAndPredicate(std::vector<Predicate*> input_preds) = 0;

  // This function iterates over all outputs of neighbouring node of the src node of the edge under merge. It checks if any of the output predicates of the neighbour node changes
  virtual Status RunFullCheckForChanges(const Edge* neighbour_edge, Predicate* new_pred, bool* is_deadness_ok) = 0;

  enum class NodeType { nSwitch, nMerge, nControlTrigger, nRecv, nGeneric };
  virtual NodeType GetNodeType(Node* n) = 0;
};

struct InputPredicateReplacementInfo {
  const Edge* e;
  Predicate* new_predicate;
};

class DeadnessAnalysisImpl : public DeadnessAnalysis {
 public:
  explicit DeadnessAnalysisImpl(const Graph* graph)
      : graph_(*graph), vlog_(VLOG_IS_ON(2)) {}
  Status Populate();
  bool HasInputsWithMismatchingDeadness(const Node& node) override;
  void Print() const override;
  // This returns an AndPredicate, otherwise it returns nullptr
  Status GetNodePredicate(const Node& node, AndPredicate** pred);
  // This function is used for creating test predicates when considering merges
  Predicate* CreateTestAndPredicate(std::vector<Predicate*> input_preds){
    return predicate_factory_.MakeAndPredicate(input_preds);
  }

  Status RunFullCheckForChanges(const Edge* neighbour_edge, Predicate* new_pred, bool* is_deadness_ok);

  //NodeType GetNodeType(Node* n);
  NodeType GetNodeType(Node* n){
  if (n->IsSwitch()) {
    return NodeType::nSwitch;
  } else if (n->IsMerge()) {
    return NodeType::nMerge;
  } else if (n->IsControlTrigger()) {
    return NodeType::nControlTrigger;
  } else if (n->IsRecv() || n->IsHostRecv()) {
    return NodeType::nRecv;
  } else {
    return NodeType::nGeneric;
  }
}

 private:
  enum class EdgeKind { kDataAndControl, kDataOnly, kControlOnly };
  std::vector<Predicate*> GetIncomingPreds(Node* n, EdgeKind edge_kind, InputPredicateReplacementInfo* replace);
  void SetPred(Node* n, int output_idx, Predicate* pred) {
    CHECK(
        predicate_map_.insert({TensorId(n->name(), output_idx), pred}).second);
  }
  void SetPred(Node* n, gtl::ArraySlice<int> output_idxs, Predicate* pred) {
    for (int output_idx : output_idxs) {
      SetPred(n, output_idx, pred);
    }
  }
  // This function dispatches the appropriate HandleX function
  // The arguments are pretty overloaded here. if replace is nullptr, then assigned_predicates is not populated.
  // If replace has a value, then one of the input predicates is replaced with the information in replace and assigned_predicates is populated, but now node predicates are not set.
  // The last element of assigned_predicates (if assigned, which it is, when replace is non-null) is the predicate for kControlSlot
  Status HandleSingleNode(Node* n, InputPredicateReplacementInfo* replace, std::vector<Predicate*>* assigned_predicates);
  // TODO: Ideally these HandleX functions should be broken into: get predicates, calculate predicates and set predicates
  Status HandleSwitch(Node* n, InputPredicateReplacementInfo* replace, std::vector<Predicate*>* assigned_predicates);
  Status HandleMerge(Node* n, InputPredicateReplacementInfo* replace, std::vector<Predicate*>* assigned_predicates);
  Status HandleRecv(Node* n, InputPredicateReplacementInfo* replace, std::vector<Predicate*>* assigned_predicates);
  Status HandleGeneric(Node* n, InputPredicateReplacementInfo* replace, std::vector<Predicate*>* assigned_predicates);
  void SetPredOrPushToVector(Node* n, bool set, int idx, Predicate* p, std::vector<Predicate*>* pred_vec);
  void SetPredOrPushToVector(Node* n, bool set, gtl::ArraySlice<int> idxs, Predicate* p, std::vector<Predicate*>* pred_vec);

  const Graph& graph_;
  gtl::FlatMap<TensorId, Predicate*, TensorId::Hasher> predicate_map_;
  PredicateFactory predicate_factory_;
  bool vlog_;
};


bool PredicateSequenceEqual(gtl::ArraySlice<Predicate*> lhs,
                            gtl::ArraySlice<Predicate*> rhs);
int64 HashPredicateSequence(Predicate::Kind kind,
                            gtl::ArraySlice<Predicate*> preds);

// Represents a logical conjunction of a set of predicates.
class AndPredicate : public Predicate {
 public:
  explicit AndPredicate(std::vector<Predicate*> operands)
      : Predicate(HashPredicateSequence(Kind::kAnd, operands)),
        operands_(std::move(operands)) {}
  string ToString() const override {
    if (operands().empty()) {
      return "#true";
    }
    std::vector<string> operands_str;
    std::transform(operands().begin(), operands().end(),
                   std::back_inserter(operands_str),
                   [](Predicate* pred) { return pred->ToString(); });
    return strings::StrCat("(", str_util::Join(operands_str, " & "), ")");
  }
  bool operator==(const Predicate& other) const override {
    return other.kind() == Kind::kAnd &&
           PredicateSequenceEqual(
               dynamic_cast<const AndPredicate&>(other).operands(), operands());
  }
  Kind kind() const override { return Kind::kAnd; }
  const tensorflow::gtl::ArraySlice<Predicate*> operands() const {
    return operands_;
  }

 private:
  std::vector<Predicate*> operands_;
};
//}


// Represents a logical disjunction of a set of predicates.
class OrPredicate : public Predicate {
 public:
  explicit OrPredicate(std::vector<Predicate*> operands)
      : Predicate(HashPredicateSequence(Kind::kOr, operands)),
        operands_(std::move(operands)) {}
  string ToString() const override {
    if (operands().empty()) {
      return "#false";
    }
    std::vector<string> operands_str;
    std::transform(operands().begin(), operands().end(),
                   std::back_inserter(operands_str),
                   [](Predicate* pred) { return pred->ToString(); });
    return strings::StrCat("(", str_util::Join(operands_str, " | "), ")");
  }
  bool operator==(const Predicate& other) const override {
    return other.kind() == Kind::kOr &&
           PredicateSequenceEqual(
               dynamic_cast<const OrPredicate&>(other).operands(), operands());
  }
  Kind kind() const override { return Kind::kOr; }
  const tensorflow::gtl::ArraySlice<Predicate*> operands() const {
    return operands_;
  }

 private:
  std::vector<Predicate*> operands_;
};
// Represents a logical negation of a set of predicates.
class NotPredicate : public Predicate {
 public:
  explicit NotPredicate(Predicate* operand)
      : Predicate(HashPredicateSequence(Kind::kNot, {operand})),
        operand_(operand) {}
  string ToString() const override {
    return strings::StrCat("~", operand()->ToString());
  }
  bool operator==(const Predicate& other) const override {
    return other.kind() == Kind::kNot &&
           *dynamic_cast<const NotPredicate&>(other).operand() == *operand();
  }
  Kind kind() const override { return Kind::kNot; }
  Predicate* operand() const { return operand_; }

 private:
  Predicate* operand_;
};

// Represents an uninterpreted symbol in a logical predicate.
//
// Two predicates are equivalent iff they are equivalent for all assignments to
// the symbols contained in them.
class SymbolPredicate : public Predicate {
 public:
  explicit SymbolPredicate(TensorId tensor_id, bool must_be_true)
      : Predicate(Hash(tensor_id, must_be_true)),
        tensor_id_(std::move(tensor_id)),
        must_be_true_(must_be_true) {}
  string ToString() const override { return tensor_id_.ToString(); }
  bool operator==(const Predicate& other) const override {
    return other.kind() == Kind::kSymbol &&
           must_be_true() ==
               dynamic_cast<const SymbolPredicate&>(other).must_be_true() &&
           dynamic_cast<const SymbolPredicate&>(other).tensor_id() ==
               tensor_id();
  }
  Kind kind() const override { return Kind::kSymbol; }
  // If `must_be_true()` is true this SymbolPredicate represents the proposition
  // "tensor_id() is live and evaluates to true".
  //
  // If `must_be_true()` is false then this SymbolPredicate represents the
  // proposition "tensor_id() is live (and may evalutate to any value)"
  TensorId tensor_id() const { return tensor_id_; }
  bool must_be_true() const { return must_be_true_; }

 private:
  TensorId tensor_id_;
  bool must_be_true_;
  static int64 Hash(const TensorId tensor_id, bool must_be_true) {
    return Hash64Combine(
        ::tensorflow::hash<bool>()(must_be_true),
        Hash64Combine(::tensorflow::hash<Predicate::Kind>()(Kind::kSymbol),
                      TensorId::Hasher{}(tensor_id)));
  }
};

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif  // NGRAPH_TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_
#endif  // NGRAPH_TF_DISABLE_DEADNESS_CHECK