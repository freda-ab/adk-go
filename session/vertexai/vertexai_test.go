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

package vertexai

import (
	"testing"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/structpb"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func TestContentRoundTrip(t *testing.T) {
	sig := []byte("opaque-thought-signature-bytes")

	event := &session.Event{
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: genai.RoleModel,
				Parts: []*genai.Part{
					{
						Text:             "thinking...",
						Thought:          true,
						ThoughtSignature: sig,
					},
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call-123",
							Name: "my_tool",
							Args: map[string]any{"key": "val"},
						},
						ThoughtSignature: sig,
					},
				},
			},
		},
	}

	pb, err := createAiplatformpbContent(event)
	if err != nil {
		t.Fatalf("createAiplatformpbContent: %v", err)
	}

	rpcResp := &aiplatformpb.SessionEvent{Content: pb}
	got := aiplatformToGenaiContent(rpcResp)

	if len(got.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Parts))
	}

	// Thought text part.
	p0 := got.Parts[0]
	if p0.Text != "thinking..." {
		t.Errorf("part[0].Text = %q, want %q", p0.Text, "thinking...")
	}
	if !p0.Thought {
		t.Error("part[0].Thought should be true")
	}
	if string(p0.ThoughtSignature) != string(sig) {
		t.Errorf("part[0].ThoughtSignature lost in round-trip")
	}

	// FunctionCall part.
	p1 := got.Parts[1]
	if p1.FunctionCall == nil {
		t.Fatal("part[1].FunctionCall is nil")
	}
	if p1.FunctionCall.ID != "call-123" {
		t.Errorf("part[1].FunctionCall.ID = %q, want %q", p1.FunctionCall.ID, "call-123")
	}
	if p1.FunctionCall.Name != "my_tool" {
		t.Errorf("part[1].FunctionCall.Name = %q, want %q", p1.FunctionCall.Name, "my_tool")
	}
	if string(p1.ThoughtSignature) != string(sig) {
		t.Errorf("part[1].ThoughtSignature lost in round-trip")
	}
}

func TestContentRoundTripFunctionResponseID(t *testing.T) {
	args, _ := structpb.NewStruct(map[string]any{"result": "ok"})
	rpcResp := &aiplatformpb.SessionEvent{
		Content: &aiplatformpb.Content{
			Role: "user",
			Parts: []*aiplatformpb.Part{
				{
					Data: &aiplatformpb.Part_FunctionResponse{
						FunctionResponse: &aiplatformpb.FunctionResponse{
							Id:       "resp-456",
							Name:     "my_tool",
							Response: args,
						},
					},
				},
			},
		},
	}

	got := aiplatformToGenaiContent(rpcResp)
	if got.Parts[0].FunctionResponse.ID != "resp-456" {
		t.Errorf("FunctionResponse.ID = %q, want %q", got.Parts[0].FunctionResponse.ID, "resp-456")
	}
}

func TestGetReasoningEngineID(t *testing.T) {
	tests := []struct {
		name             string
		existingEngineID string // Field: c.reasoningEngine
		inputAppName     string // Argument: appName
		expectedID       string
		expectError      bool
	}{
		{
			name:             "Client already has engine ID configured",
			existingEngineID: "999",
			inputAppName:     "irrelevant-input",
			expectedID:       "999",
			expectError:      false,
		},
		{
			name:             "Input is a direct numeric ID",
			existingEngineID: "",
			inputAppName:     "123456",
			expectedID:       "123456",
			expectError:      false,
		},
		{
			name:             "Input is a valid full resource path",
			existingEngineID: "",
			inputAppName:     "projects/my-project/locations/us-central1/reasoningEngines/555123",
			expectedID:       "555123",
			expectError:      false,
		},
		{
			name:             "Input is valid path with dashes and underscores in project/location",
			existingEngineID: "",
			inputAppName:     "projects/my_project-1/locations/us_central-1/reasoningEngines/888",
			expectedID:       "888",
			expectError:      false,
		},
		{
			name:             "Input is malformed (ID is not numeric)",
			existingEngineID: "",
			inputAppName:     "projects/proj/locations/loc/reasoningEngines/abc",
			expectedID:       "",
			expectError:      true,
		},
		{
			name:             "Input is malformed (missing path components)",
			existingEngineID: "",
			inputAppName:     "locations/us-central1/reasoningEngines/123",
			expectedID:       "",
			expectError:      true,
		},
		{
			name:             "Input is random text",
			existingEngineID: "",
			inputAppName:     "some-random-app-name",
			expectedID:       "",
			expectError:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup the client with the test case state
			c := &vertexAiClient{
				reasoningEngine: tt.existingEngineID,
			}

			// Execute
			got, err := c.getReasoningEngineID(tt.inputAppName)

			// Check Error Expectation
			if (err != nil) != tt.expectError {
				t.Errorf("getReasoningEngineID() error = %v, expectError %v", err, tt.expectError)
				return
			}

			// Check Returned Value
			if got != tt.expectedID {
				t.Errorf("getReasoningEngineID() got = %v, want %v", got, tt.expectedID)
			}
		})
	}
}
