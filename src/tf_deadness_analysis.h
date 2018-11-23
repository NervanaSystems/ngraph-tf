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

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
#ifndef NGRAPH_TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_
#define NGRAPH_TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace ngraph_bridge {

class Predicate;
class AndPredicate;

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

  virtual Status GetEdgePredicate(const Edge* edge, Predicate** pred) = 0;
};

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

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif  // NGRAPH_TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_
#endif  // NGRAPH_TF_DISABLE_DEADNESS_CHECK