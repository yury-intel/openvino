//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "constant_folding.hpp"
#include <ngraph/rt_info.hpp>

using namespace std;
using namespace ngraph;

bool ngraph::pass::revalidate_and_ensure_static(shared_ptr<Node> n)
{
    n->revalidate_and_infer_types();
    for (auto& o : n->outputs())
    {
        if (o.get_partial_shape().is_dynamic() || o.get_element_type().is_dynamic())
        {
            return false;
        }
    }
    return true;
}

bool ngraph::pass::ConstantFolding::cf_is_disabled(const std::shared_ptr<Node>& node)
{
    auto& rt_info = node->get_rt_info();
    return rt_info.count("DISABLED_CONSTANT_FOLDING") != 0;
}

void ngraph::pass::ConstantFolding::copy_runtime_info_to_target_inputs(
    const std::shared_ptr<Node>& node, const Output<Node>& replacement)
{
    for (auto& input : replacement.get_target_inputs())
    {
        auto consumer = input.get_node()->shared_from_this();
        copy_runtime_info({node, consumer}, consumer);
    }
}

void ngraph::pass::ConstantFolding::construct_constant_default()
{
    m_matchers.push_back(std::make_shared<MatcherPass>(
        "Constant folding defaults",
        nullptr,
        [=](const std::shared_ptr<Node>& node) -> bool {
            OutputVector replacements(node->get_output_size());
            if (!node->constant_fold(replacements, node->input_values()))
            {
                return false;
            }
            NGRAPH_CHECK(replacements.size() == node->get_output_size(),
                         "constant_fold_default returned incorrect number of replacements for ",
                         node);
            bool result{false};
            for (size_t i = 0; i < replacements.size(); ++i)
            {
                auto node_output = node->output(i);
                auto replacement = replacements.at(i);
                if (replacement.get_node_shared_ptr() && (node_output != replacement))
                {
                    if (replacements.size() == 1)
                    {
                        replacement.get_node_shared_ptr()->set_friendly_name(
                            node->get_friendly_name());
                    }
                    else
                    {
                        replacement.get_node_shared_ptr()->set_friendly_name(
                            node->get_friendly_name() + "." + std::to_string(i));
                    }
                    node_output.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(node, replacement);
                    result = true;
                }
            }
            return result;
        },
        PassProperty::CHANGE_DYNAMIC_STATE));
}
