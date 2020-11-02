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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>

#include "itt.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

pass::Manager::Manager()
    : m_visualize(getenv_bool("NGRAPH_ENABLE_VISUALIZE_TRACING"))
    , m_pass_config(std::make_shared<PassConfig>())
{
}

pass::Manager::~Manager()
{
}

void pass::Manager::run_passes(shared_ptr<Function> func)
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraph, "pass::Manager::run_passes");

    static bool profile_enabled = getenv_bool("NGRAPH_PROFILE_PASS_ENABLE");

    size_t index = 0;
    stopwatch pass_timer;
    stopwatch overall_timer;
    overall_timer.start();
    bool function_changed = false;
    for (auto& pass : m_pass_list)
    {
        if (m_pass_config->is_disabled(pass->get_type_info()))
        {
            NGRAPH_DEBUG << "Pass " << pass->get_name() << " is disabled";
            continue;
        }

        pass_timer.start();

        NGRAPH_SUPPRESS_DEPRECATED_START
        if (auto matcher_pass = dynamic_pointer_cast<MatcherPass>(pass))
        {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (matcher_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) &&
                func->is_dynamic())
            {
                NGRAPH_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                             << "function is dynamic. Skipping this transformation";
                continue;
            }
            // GraphRewrite is a temporary container for MatcherPass to make execution
            // on on entire ngraph::Function
            function_changed = GraphRewrite(matcher_pass).run_on_function(func);
        }
        else if (auto function_pass = dynamic_pointer_cast<FunctionPass>(pass))
        {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (function_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) &&
                func->is_dynamic())
            {
                NGRAPH_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                             << "function is dynamic. Skipping this transformation";
                continue;
            }

            if (dynamic_pointer_cast<Validate>(pass))
            {
                if (function_changed)
                {
                    function_pass->run_on_function(func);
                    function_changed = false;
                }
            }
            else
            {
                function_changed = function_pass->run_on_function(func);
            }
        }
        else if (auto node_pass = dynamic_pointer_cast<NodePass>(pass))
        {
            if (node_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && func->is_dynamic())
            {
                NGRAPH_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                             << "function is dynamic. Skipping this transformation";
                continue;
            }
            for (shared_ptr<Node> n : func->get_ops())
            {
                function_changed |= node_pass->run_on_node(n);
            }
        }
        NGRAPH_SUPPRESS_DEPRECATED_END

        if (m_visualize)
        {
            // visualizations and serializations will be named after the outermost function
            const size_t num_digits_in_pass_index = 3;
            std::string index_str = std::to_string(index);
            index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
            auto base_filename = func->get_name() + std::string("_") + index_str +
                                 std::string("_") + pass->get_name();

            if (m_visualize)
            {
                static const string format = getenv_string("NGRAPH_VISUALIZE_TRACING_FORMAT");
                auto file_ext = format.empty() ? "svg" : format;
                pass::VisualizeTree vt(base_filename + std::string(".") + file_ext);
                vt.run_on_function(func);
            }
        }
        index++;
        pass_timer.stop();
        if (profile_enabled)
        {
            cout << setw(7) << pass_timer.get_milliseconds() << "ms " << pass->get_name() << "\n";
        }
    }
    if (profile_enabled)
    {
        cout << "passes done in " << overall_timer.get_milliseconds() << "ms\n";
    }
}
