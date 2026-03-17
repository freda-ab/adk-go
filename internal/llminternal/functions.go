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
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/converters"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// generateRequestConfirmationEvent creates a new Event containing
// adk_request_confirmation function calls based on the requested confirmations.
// NOTE: The trigger for this in ADK Go is usually a tool.Context.RequestConfirmation call,
// not parsing a function_response_event like in the Python example.
// This function assumes you have a list of confirmations to process.
func generateRequestConfirmationEvent(
	invocationContext agent.InvocationContext,
	functionCallEvent *session.Event,
	functionResponseEvent *session.Event,
) (*session.Event, error) {
	if functionResponseEvent == nil || len(functionResponseEvent.Actions.RequestedToolConfirmations) == 0 {
		return nil, nil
	}
	if functionCallEvent == nil || functionCallEvent.Content == nil {
		return nil, nil
	}

	parts := []*genai.Part{}
	longRunningToolIDs := []string{}

	// Index the original Parts by function call ID so we can carry over
	// ThoughtSignature. Gemini thinking models require every model-role
	// function call Part to include its thought signature; without it the
	// API returns INVALID_ARGUMENT.
	type originalCall struct {
		call             *genai.FunctionCall
		thoughtSignature []byte
	}
	originals := make(map[string]originalCall, len(functionCallEvent.Content.Parts))
	for _, p := range functionCallEvent.Content.Parts {
		if p.FunctionCall != nil {
			originals[p.FunctionCall.ID] = originalCall{
				call:             p.FunctionCall,
				thoughtSignature: p.ThoughtSignature,
			}
		}
	}

	for funcID, confirmation := range functionResponseEvent.Actions.RequestedToolConfirmations {
		orig, ok := originals[funcID]
		if !ok || orig.call == nil {
			continue
		}

		originalCallMap, err := converters.ToMapStructure(orig.call)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize original function call: %w", err)
		}
		confirmationMap, err := converters.ToMapStructure(confirmation)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize tool confirmation: %w", err)
		}
		args := map[string]any{
			"originalFunctionCall": originalCallMap,
			"toolConfirmation":     confirmationMap,
		}

		requestConfirmationFC := &genai.FunctionCall{
			ID:   utils.GenerateFunctionCallID(),
			Name: toolconfirmation.FunctionCallName,
			Args: args,
		}

		parts = append(parts, &genai.Part{
			FunctionCall:     requestConfirmationFC,
			ThoughtSignature: orig.thoughtSignature,
		})
		longRunningToolIDs = append(longRunningToolIDs, requestConfirmationFC.ID)
	}

	if len(parts) == 0 {
		return nil, nil
	}

	ev := session.NewEvent(invocationContext.InvocationID())
	ev.Author = invocationContext.Agent().Name()
	ev.Branch = invocationContext.Branch()
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Parts: parts,
			Role:  genai.RoleModel,
		},
	}
	ev.LongRunningToolIDs = longRunningToolIDs
	return ev, nil
}
