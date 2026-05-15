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
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func TestEventsForInvocation(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	svc := session.InMemoryService()
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	sess := resp.Session

	events := []*session.Event{
		{InvocationID: "inv-1", Author: "user"},
		{InvocationID: "inv-1", Author: "agent"},
		{InvocationID: "inv-2", Author: "user"},
		{InvocationID: "inv-2", Author: "agent"},
		{InvocationID: "inv-1", Author: "agent"},
	}
	for _, e := range events {
		if err := svc.AppendEvent(ctx, sess, e); err != nil {
			t.Fatal(err)
		}
	}

	got := eventsForInvocation(sess.Events(), "inv-1")
	if len(got) != 3 {
		t.Errorf("got %d events, want 3", len(got))
	}
	got = eventsForInvocation(sess.Events(), "inv-2")
	if len(got) != 2 {
		t.Errorf("got %d events, want 2", len(got))
	}
	got = eventsForInvocation(sess.Events(), "inv-3")
	if len(got) != 0 {
		t.Errorf("got %d events, want 0", len(got))
	}
}

func TestEndOfAgents(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		events []*session.Event
		want   map[string]bool
	}{
		{
			name:   "empty",
			events: nil,
			want:   map[string]bool{},
		},
		{
			name: "agent finished with text response",
			events: []*session.Event{
				{Author: "user"},
				{
					Author: "my_agent",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "model",
							Parts: []*genai.Part{{Text: "done"}},
						},
					},
				},
			},
			want: map[string]bool{"my_agent": true},
		},
		{
			name: "agent has pending function calls",
			events: []*session.Event{
				{Author: "user"},
				{
					Author: "my_agent",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "tool1"},
							}},
						},
					},
				},
			},
			want: map[string]bool{"my_agent": false},
		},
		{
			name: "agent finished after tool call cycle",
			events: []*session.Event{
				{Author: "user"},
				{
					Author: "my_agent",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "tool1"},
							}},
						},
					},
				},
				{
					Author: "my_agent",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: "user",
							Parts: []*genai.Part{{
								FunctionResponse: &genai.FunctionResponse{ID: "fc-1", Name: "tool1"},
							}},
						},
					},
				},
				{
					Author: "my_agent",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "model",
							Parts: []*genai.Part{{Text: "final answer"}},
						},
					},
				},
			},
			want: map[string]bool{"my_agent": true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := endOfAgents(tt.events)
			for k, v := range tt.want {
				if got[k] != v {
					t.Errorf("endOfAgents[%q] = %v, want %v", k, got[k], v)
				}
			}
			if len(got) != len(tt.want) {
				t.Errorf("endOfAgents has %d entries, want %d", len(got), len(tt.want))
			}
		})
	}
}

func TestResumeShortCircuit(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	svc := session.InMemoryService()

	agentCalled := false
	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				agentCalled = true
			}
		},
	}))

	r, err := New(Config{
		AppName:        "app",
		Agent:          testAgent,
		SessionService: svc,
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	// Seed session with events for invocation "inv-1" where agent finished
	seedEvents := []*session.Event{
		{
			InvocationID: "inv-1",
			Author:       "user",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		},
		{
			InvocationID: "inv-1",
			Author:       "test_agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "done"}}},
			},
		},
	}
	for _, e := range seedEvents {
		if err := svc.AppendEvent(ctx, resp.Session, e); err != nil {
			t.Fatal(err)
		}
	}

	var yielded []*session.Event
	for ev, err := range r.Run(ctx, "u", "s", nil, agent.RunConfig{},
		WithInvocationID("inv-1"), WithResume()) {
		if err != nil {
			t.Fatal(err)
		}
		if ev != nil {
			yielded = append(yielded, ev)
		}
	}

	if agentCalled {
		t.Error("agent should not have been called on short-circuit")
	}
	if len(yielded) != 0 {
		t.Errorf("expected 0 yielded events, got %d", len(yielded))
	}
}

func TestResumeSkipsUserMessageAppend(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	svc := session.InMemoryService()

	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// Agent does nothing
			}
		},
	}))

	r, err := New(Config{
		AppName:        "app",
		Agent:          testAgent,
		SessionService: svc,
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	// Seed: user message + agent has pending function call (not finished)
	seedEvents := []*session.Event{
		{
			InvocationID: "inv-1",
			Author:       "user",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		},
		{
			InvocationID: "inv-1",
			Author:       "test_agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "some_tool"},
					}},
				},
			},
		},
	}
	for _, e := range seedEvents {
		if err := svc.AppendEvent(ctx, resp.Session, e); err != nil {
			t.Fatal(err)
		}
	}

	eventCountBefore := resp.Session.Events().Len()

	// Resume with a user message — it should NOT be appended
	for _, err := range r.Run(ctx, "u", "s",
		&genai.Content{Parts: []*genai.Part{{Text: "retry"}}},
		agent.RunConfig{},
		WithInvocationID("inv-1"), WithResume()) {
		if err != nil {
			t.Fatal(err)
		}
	}

	// Count user-authored events
	userEvents := 0
	for e := range resp.Session.Events().All() {
		if e.Author == "user" {
			userEvents++
		}
	}

	if userEvents != 1 {
		t.Errorf("expected 1 user event (original), got %d", userEvents)
	}

	// Total events should not have grown by a user message
	// (agent might add events, but the user message should not be re-added)
	if resp.Session.Events().Len() < eventCountBefore {
		t.Error("event count should not have decreased")
	}
}

func TestResumeWithoutFlagDoesNotResume(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	svc := session.InMemoryService()

	agentCalled := false
	testAgent := must(agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				agentCalled = true
			}
		},
	}))

	r, err := New(Config{
		AppName:        "app",
		Agent:          testAgent,
		SessionService: svc,
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	// Seed: agent already finished
	seedEvents := []*session.Event{
		{
			InvocationID: "inv-1",
			Author:       "user",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
			},
		},
		{
			InvocationID: "inv-1",
			Author:       "test_agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "done"}}},
			},
		},
	}
	for _, e := range seedEvents {
		if err := svc.AppendEvent(ctx, resp.Session, e); err != nil {
			t.Fatal(err)
		}
	}

	// WithInvocationID but WITHOUT WithResume — should run agent normally
	for _, err := range r.Run(ctx, "u", "s", nil, agent.RunConfig{},
		WithInvocationID("inv-1")) {
		if err != nil {
			t.Fatal(err)
		}
	}

	if !agentCalled {
		t.Error("agent should have been called when WithResume is not set")
	}
}

func TestLastActiveAgent(t *testing.T) {
	t.Parallel()

	sub := must(agent.New(agent.Config{Name: "sub_agent"}))
	root := must(agent.New(agent.Config{
		Name:      "root_agent",
		SubAgents: []agent.Agent{sub},
	}))

	tests := []struct {
		name   string
		events []*session.Event
		want   string
	}{
		{
			name:   "empty events",
			events: nil,
			want:   "",
		},
		{
			name: "only user events",
			events: []*session.Event{
				{Author: "user"},
			},
			want: "",
		},
		{
			name: "root agent active",
			events: []*session.Event{
				{Author: "user"},
				{Author: "root_agent"},
			},
			want: "root_agent",
		},
		{
			name: "sub agent was last active",
			events: []*session.Event{
				{Author: "user"},
				{Author: "root_agent"},
				{Author: "sub_agent"},
			},
			want: "sub_agent",
		},
		{
			name: "unknown agent skipped",
			events: []*session.Event{
				{Author: "root_agent"},
				{Author: "unknown_agent"},
			},
			want: "root_agent",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := lastActiveAgent(root, tt.events)
			if tt.want == "" {
				if got != nil {
					t.Errorf("expected nil, got %q", got.Name())
				}
			} else if got == nil {
				t.Errorf("expected %q, got nil", tt.want)
			} else if got.Name() != tt.want {
				t.Errorf("expected %q, got %q", tt.want, got.Name())
			}
		})
	}
}

func TestResumeUsesInvocationAgent(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	svc := session.InMemoryService()

	var ranAgent string
	sub := must(agent.New(agent.Config{
		Name: "sub_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ranAgent = "sub_agent"
			}
		},
	}))
	root := must(agent.New(agent.Config{
		Name:      "root_agent",
		SubAgents: []agent.Agent{sub},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ranAgent = "root_agent"
			}
		},
	}))

	r, err := New(Config{
		AppName:        "app",
		Agent:          root,
		SessionService: svc,
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	// Invocation inv-1: sub_agent was active but not finished
	// Then a newer invocation inv-2 completed with root_agent
	seedEvents := []*session.Event{
		{InvocationID: "inv-1", Author: "user", LLMResponse: model.LLMResponse{
			Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "go"}}},
		}},
		{InvocationID: "inv-1", Author: "sub_agent", LLMResponse: model.LLMResponse{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "tool1"},
			}}},
		}},
		{InvocationID: "inv-2", Author: "user", LLMResponse: model.LLMResponse{
			Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "new msg"}}},
		}},
		{InvocationID: "inv-2", Author: "root_agent", LLMResponse: model.LLMResponse{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "done"}}},
		}},
	}
	for _, e := range seedEvents {
		if err := svc.AppendEvent(ctx, resp.Session, e); err != nil {
			t.Fatal(err)
		}
	}

	// Resume inv-1 — should run sub_agent (from inv-1), not root_agent (from inv-2)
	for _, err := range r.Run(ctx, "u", "s", nil, agent.RunConfig{},
		WithInvocationID("inv-1"), WithResume()) {
		if err != nil {
			t.Fatal(err)
		}
	}

	if ranAgent != "sub_agent" {
		t.Errorf("expected sub_agent to run on resume, got %q", ranAgent)
	}
}
