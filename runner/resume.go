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

package runner

import (
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// eventsForInvocation returns events matching the given invocation ID, in order.
func eventsForInvocation(events session.Events, invocationID string) []*session.Event {
	var result []*session.Event
	for e := range events.All() {
		if e.InvocationID == invocationID {
			result = append(result, e)
		}
	}
	return result
}

// lastActiveAgent finds the agent that was last active in the invocation by
// scanning events backwards for the last non-user author. This is used instead
// of findAgentToRun during resume so the agent selection is based on the
// invocation's own events, not the full session which may contain newer
// invocations.
func lastActiveAgent(root agent.Agent, invocationEvents []*session.Event) agent.Agent {
	for i := len(invocationEvents) - 1; i >= 0; i-- {
		ev := invocationEvents[i]
		if ev.Author == "" || ev.Author == "user" {
			continue
		}
		if a := root.FindAgent(ev.Author); a != nil {
			return a
		}
	}
	return nil
}

// endOfAgents replays invocation events to determine which agents already
// finished. An agent is "done" if its last event is a final response.
func endOfAgents(invocationEvents []*session.Event) map[string]bool {
	done := make(map[string]bool)
	for _, ev := range invocationEvents {
		if ev.Author == "" || ev.Author == "user" {
			continue
		}
		if ev.IsFinalResponse() {
			done[ev.Author] = true
		} else {
			done[ev.Author] = false
		}
	}
	return done
}
