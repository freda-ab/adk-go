// Copyright 2026 Google LLC
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

package llminternal_test

import (
	"context"
	"iter"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/adk/tool/functiontool"
)

type SleepArgs struct {
	DurationMS int `json:"duration_ms"`
}
type SleepResult struct {
	Success bool `json:"success"`
}

type NavigateArgs struct {
	Path string `json:"path"`
}

type NavigateResult struct {
	Status string         `json:"status"`
	Results map[string]any `json:"results,omitempty"`
}

func sleepFunc(ctx tool.Context, input SleepArgs) (SleepResult, error) {
	time.Sleep(time.Duration(input.DurationMS) * time.Millisecond)
	return SleepResult{Success: true}, nil
}

func navigateFunc(ctx tool.Context, input NavigateArgs) (NavigateResult, error) {
	if ctx.ToolConfirmation() == nil {
		if err := ctx.RequestConfirmation("Navigate to route", input); err != nil {
			return NavigateResult{}, err
		}
		return NavigateResult{Status: "AWAITING_USER_INPUT"}, nil
	}
	return NavigateResult{
		Status:  "SUCCESS",
		Results: map[string]any{"path": input.Path},
	}, nil
}

// mockModel is a simple mock model that returns parallel tool calls.
type mockModel struct {
	model.LLM
	Calls int
}

func (m *mockModel) Name() string {
	return "mock-model"
}

func (m *mockModel) GenerateContent(ctx context.Context, req *model.LLMRequest, useStream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.Calls++
		if m.Calls > 1 {
			// Second call should be the final response after tool execution.
			// Or we just return a final response if we don't want to loop.
			// For this test, we just need to trigger the tool calls once.
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						genai.NewPartFromText("I am done."),
					},
					Role: "model",
				},
				Partial: false,
			}, nil)
			return
		}

		// First call returns parallel tool calls.
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_1",
							Name: "sleep",
							Args: map[string]any{"duration_ms": 500},
						},
					},
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_2",
							Name: "sleep",
							Args: map[string]any{"duration_ms": 500},
						},
					},
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_3",
							Name: "sleep",
							Args: map[string]any{"duration_ms": 500},
						},
					},
				},
				Role: "model",
			},
			Partial: false,
		}, nil)
	}
}

func TestHandleFunctionCallsAsync(t *testing.T) {
	sleepTool, err := functiontool.New(functiontool.Config{
		Name:        "sleep",
		Description: "sleeps for a duration",
	}, sleepFunc)
	if err != nil {
		t.Fatal(err)
	}

	model := &mockModel{}

	a, err := llmagent.New(llmagent.Config{
		Name:        "tester",
		Description: "Tester agent",
		Instruction: "You are a tester agent.",
		Model:       model,
		Tools: []tool.Tool{
			sleepTool,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	sessionService := session.InMemoryService()
	_, err = sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	})
	if err != nil {
		t.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		Agent:          a,
		SessionService: sessionService,
		AppName:        "testApp",
	})
	if err != nil {
		t.Fatal(err)
	}

	startTime := time.Now()

	it := r.Run(t.Context(), "testUser", "testSession", &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("Test sleep"),
		},
		Role: "user",
	}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})

	events := []*session.Event{}
	for ev, err := range it {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	if len(events) != 3 {
		t.Errorf("Expected 3 events, got %d", len(events))
	}

	elapsed := time.Since(startTime)
	t.Logf("Elapsed time: %v", elapsed)

	if len(events[0].Content.Parts) != 3 {
		t.Errorf("Expected first event to have 3 function calls, got %d", len(events[0].Content.Parts))
	}
	if len(events[1].Content.Parts) != 3 {
		t.Errorf("Expected second event to have 3 function responses, got %d", len(events[1].Content.Parts))
	}
	if len(events[2].Content.Parts) != 1 {
		t.Errorf("Expected third event to have 1 text part got %d", len(events[2].Content.Parts))
	}

	// Since we are calling sleep 3 times for 500ms each, synchronous execution would take
	// ~1500ms, while asynchronous execution should take ~500ms.
	// We assert that the time is significantly less than 1500ms to verify async.
	// We also assert it's at least 500ms.

	if elapsed < 500*time.Millisecond {
		t.Errorf("Elapsed time %v is less than expected 500ms", elapsed)
	}

	if elapsed > 1000*time.Millisecond {
		t.Errorf("Elapsed time %v is greater than expected 1000ms for async execution", elapsed)
	}
}

type confirmationOrderingModel struct {
	model.LLM
	calls int
}

func (m *confirmationOrderingModel) Name() string {
	return "confirmation-ordering-model"
}

func (m *confirmationOrderingModel) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	useStream bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		if m.calls > 1 {
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						genai.NewPartFromText("done"),
					},
					Role: "model",
				},
				Partial: false,
			}, nil)
			return
		}

		yield(&model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call-nav",
							Name: "navigate",
							Args: map[string]any{"path": "/vendors/example"},
						},
					},
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call-sleep",
							Name: "sleep",
							Args: map[string]any{"duration_ms": 1},
						},
					},
				},
				Role: "model",
			},
			Partial: false,
		}, nil)
	}
}

func TestRunnerPersistsFunctionResponsesBeforeConfirmationWhenConsumerStops(t *testing.T) {
	navigateTool, err := functiontool.New(functiontool.Config{
		Name:        "navigate",
		Description: "navigates and requests confirmation",
	}, navigateFunc)
	if err != nil {
		t.Fatal(err)
	}

	sleepTool, err := functiontool.New(functiontool.Config{
		Name:        "sleep",
		Description: "sleeps for a duration",
	}, sleepFunc)
	if err != nil {
		t.Fatal(err)
	}

	model := &confirmationOrderingModel{}
	a, err := llmagent.New(llmagent.Config{
		Name:        "tester",
		Description: "Tester agent",
		Instruction: "You are a tester agent.",
		Model:       model,
		Tools: []tool.Tool{
			navigateTool,
			sleepTool,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	sessionService := session.InMemoryService()
	_, err = sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	})
	if err != nil {
		t.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		Agent:          a,
		SessionService: sessionService,
		AppName:        "testApp",
	})
	if err != nil {
		t.Fatal(err)
	}

	it := r.Run(t.Context(), "testUser", "testSession", &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("Set up the vendor"),
		},
		Role: "user",
	}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})

	for ev, err := range it {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Content == nil {
			continue
		}
		stop := false
		for _, part := range ev.Content.Parts {
			if part.FunctionCall != nil && part.FunctionCall.Name == toolconfirmation.FunctionCallName {
				stop = true
				break
			}
		}
		if stop {
			break
		}
	}

	resp, err := sessionService.Get(t.Context(), &session.GetRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	})
	if err != nil {
		t.Fatal(err)
	}

	var events []*session.Event
	for ev := range resp.Session.Events().All() {
		events = append(events, ev)
	}

	responseIndex := -1
	confirmationIndex := -1
	for i, ev := range events {
		if ev.Content == nil {
			continue
		}
		hasOnlyResponses := len(ev.Content.Parts) > 0
		hasConfirmation := false
		for _, part := range ev.Content.Parts {
			if part.FunctionResponse == nil {
				hasOnlyResponses = false
			}
			if part.FunctionCall != nil && part.FunctionCall.Name == toolconfirmation.FunctionCallName {
				hasConfirmation = true
			}
		}
		if hasOnlyResponses && responseIndex == -1 {
			responseIndex = i
		}
		if hasConfirmation && confirmationIndex == -1 {
			confirmationIndex = i
		}
	}

	if responseIndex == -1 {
		t.Fatalf("expected a persisted function-response event, got %d events", len(events))
	}
	if confirmationIndex == -1 {
		t.Fatalf("expected a persisted confirmation event, got %d events", len(events))
	}
	if responseIndex >= confirmationIndex {
		t.Fatalf("expected function responses to persist before confirmation, got responseIndex=%d confirmationIndex=%d", responseIndex, confirmationIndex)
	}

	if len(events[responseIndex].Content.Parts) != 2 {
		t.Fatalf("expected persisted function-response event to contain 2 parts, got %#v", events[responseIndex].Content)
	}
	for i, part := range events[responseIndex].Content.Parts {
		if part.FunctionResponse == nil {
			t.Fatalf("events[%d].parts[%d]: expected function response, got %#v", responseIndex, i, part)
		}
	}
}
