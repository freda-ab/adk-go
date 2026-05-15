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
	"context"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

func createResumeCtx(t *testing.T, agnt agent.Agent, sess session.Session, invocationID string, resumable bool) agent.InvocationContext {
	t.Helper()
	return icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
		Agent:        agnt,
		Session:      sess,
		InvocationID: invocationID,
		Resumable:    resumable,
	})
}

func TestMaybeResumeTools_NoEvents(t *testing.T) {
	t.Parallel()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	agnt, _, err := newTestLLMAgent()
	if err != nil {
		t.Fatal(err)
	}
	ctx := createResumeCtx(t, agnt, resp.Session, "inv-1", true)

	f := &Flow{}
	ev, err := f.maybeResumeTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ev != nil {
		t.Error("expected nil event when no events exist")
	}
}

func TestMaybeResumeTools_LastEventHasNoFunctionCalls(t *testing.T) {
	t.Parallel()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	err = svc.AppendEvent(t.Context(), resp.Session, &session.Event{
		InvocationID: "inv-1",
		Author:       "agent",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: "hello"}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	agnt, _, err := newTestLLMAgent()
	if err != nil {
		t.Fatal(err)
	}
	ctx := createResumeCtx(t, agnt, resp.Session, "inv-1", true)

	f := &Flow{}
	ev, err := f.maybeResumeTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ev != nil {
		t.Error("expected nil event when last event has no function calls")
	}
}

func TestMaybeResumeTools_ReExecutesUnrespondedCalls(t *testing.T) {
	t.Parallel()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	// Last event has a function call with no response
	err = svc.AppendEvent(t.Context(), resp.Session, &session.Event{
		InvocationID: "inv-1",
		Author:       "agent",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   "fc-1",
						Name: "test_tool",
						Args: map[string]any{"key": "value"},
					},
				}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	toolCalled := false
	testTool := &mockFunctionTool{
		name: "test_tool",
		runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
			toolCalled = true
			return map[string]any{"result": "success"}, nil
		},
	}

	agnt, _, err := newTestLLMAgent()
	if err != nil {
		t.Fatal(err)
	}
	ctx := createResumeCtx(t, agnt, resp.Session, "inv-1", true)

	f := &Flow{
		Tools: []tool.Tool{testTool},
	}

	ev, err := f.maybeResumeTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !toolCalled {
		t.Error("expected tool to be called on resume")
	}
	if ev == nil {
		t.Fatal("expected non-nil event")
	}
	if ev.Content == nil || len(ev.Content.Parts) == 0 {
		t.Fatal("expected event with function response parts")
	}
	if ev.Content.Parts[0].FunctionResponse == nil {
		t.Fatal("expected function response part")
	}
	if ev.Content.Parts[0].FunctionResponse.ID != "fc-1" {
		t.Errorf("function response ID = %q, want %q", ev.Content.Parts[0].FunctionResponse.ID, "fc-1")
	}
}

func TestMaybeResumeTools_IgnoresOtherInvocationEvents(t *testing.T) {
	t.Parallel()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	// Event from a different invocation has function calls
	err = svc.AppendEvent(t.Context(), resp.Session, &session.Event{
		InvocationID: "inv-OTHER",
		Author:       "agent",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "test_tool"},
				}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Our invocation has only a text event
	err = svc.AppendEvent(t.Context(), resp.Session, &session.Event{
		InvocationID: "inv-1",
		Author:       "agent",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: "hello"}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	agnt, _, err := newTestLLMAgent()
	if err != nil {
		t.Fatal(err)
	}
	ctx := createResumeCtx(t, agnt, resp.Session, "inv-1", true)

	f := &Flow{}
	ev, err := f.maybeResumeTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ev != nil {
		t.Error("should not resume tools from other invocation")
	}
}

func TestShouldStayPaused_NoPause(t *testing.T) {
	t.Parallel()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	err = svc.AppendEvent(t.Context(), resp.Session, &session.Event{
		InvocationID: "inv-1",
		Author:       "agent",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: "hello"}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	agnt, _, err := newTestLLMAgent()
	if err != nil {
		t.Fatal(err)
	}
	ctx := createResumeCtx(t, agnt, resp.Session, "inv-1", true)

	f := &Flow{}
	if f.shouldStayPaused(ctx) {
		t.Error("should not be paused when no long-running tool IDs")
	}
}

func TestShouldStayPaused_LongRunningToolPending(t *testing.T) {
	t.Parallel()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}

	err = svc.AppendEvent(t.Context(), resp.Session, &session.Event{
		InvocationID:   "inv-1",
		Author:         "agent",
		LongRunningToolIDs: []string{"fc-1"},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "slow_tool"},
				}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	agnt, _, err := newTestLLMAgent()
	if err != nil {
		t.Fatal(err)
	}
	ctx := createResumeCtx(t, agnt, resp.Session, "inv-1", true)

	f := &Flow{}
	if !f.shouldStayPaused(ctx) {
		t.Error("should be paused when long-running tool ID matches a function call")
	}
}

func TestAnnotateResumeContents_AddsSystemInstruction(t *testing.T) {
	t.Parallel()
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hello"}}},
		},
	}
	annotateResumeContents(req)

	if req.Config == nil || req.Config.SystemInstruction == nil {
		t.Fatal("expected system instruction to be set")
	}
	parts := req.Config.SystemInstruction.Parts
	if len(parts) == 0 || parts[len(parts)-1].Text == "" {
		t.Fatal("expected non-empty system instruction text")
	}
}

func TestAnnotateResumeContents_EmptyRequest(t *testing.T) {
	t.Parallel()
	req := &model.LLMRequest{}
	annotateResumeContents(req)
	if req.Config == nil || req.Config.SystemInstruction == nil {
		t.Fatal("expected system instruction even with empty contents")
	}
}

func newTestLLMAgent() (agent.Agent, []tool.Tool, error) {
	agnt, err := agent.New(agent.Config{
		Name: "test_agent",
	})
	return agnt, nil, err
}
