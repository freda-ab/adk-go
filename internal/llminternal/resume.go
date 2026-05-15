// Copyright 2025 Google LLC
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

package llminternal

import (
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// maybeResumeTools checks if the last event on the current branch has
// unresponded function calls. If so, it re-executes them without calling the
// LLM.
func (f *Flow) maybeResumeTools(ctx agent.InvocationContext) (*session.Event, error) {
	events := invocationBranchEvents(ctx)
	if len(events) == 0 {
		return nil, nil
	}

	last := events[len(events)-1]
	fnCalls := utils.FunctionCalls(last.Content)
	if len(fnCalls) == 0 {
		return nil, nil
	}

	toolsDict := make(map[string]tool.Tool, len(f.Tools))
	for _, t := range f.Tools {
		toolsDict[t.Name()] = t
	}

	return f.handleFunctionCalls(ctx, toolsDict, &last.LLMResponse, nil)
}

// shouldStayPaused returns true if the last 1-2 events on the current branch
// contain long-running tool calls that haven't been resolved.
func (f *Flow) shouldStayPaused(ctx agent.InvocationContext) bool {
	events := invocationBranchEvents(ctx)
	start := len(events) - 2
	if start < 0 {
		start = 0
	}
	for _, ev := range events[start:] {
		if len(ev.LongRunningToolIDs) == 0 {
			continue
		}
		longRunning := make(map[string]struct{}, len(ev.LongRunningToolIDs))
		for _, id := range ev.LongRunningToolIDs {
			longRunning[id] = struct{}{}
		}
		for _, fc := range utils.FunctionCalls(ev.Content) {
			if _, ok := longRunning[fc.ID]; ok {
				return true
			}
		}
	}
	return false
}

// invocationBranchEvents returns events for the current invocation and branch.
func invocationBranchEvents(ctx agent.InvocationContext) []*session.Event {
	if ctx.Session() == nil {
		return nil
	}
	var events []*session.Event
	for e := range ctx.Session().Events().All() {
		if e.InvocationID == ctx.InvocationID() && eventBelongsToBranch(ctx.Branch(), e) {
			events = append(events, e)
		}
	}
	return events
}

// annotateResumeContents adds a system instruction hint when resuming, so the
// model knows it is continuing from a prior attempt rather than starting fresh.
func annotateResumeContents(req *model.LLMRequest) {
	utils.AppendInstructions(req,
		"This is a resumed invocation. The conversation history contains tool calls and responses from a prior attempt. Continue where you left off.",
	)
}
