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
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type SumArgs struct {
	A int `json:"a"` // an integer to sum
	B int `json:"b"` // another integer to sum
}
type SumResult struct {
	Sum int `json:"sum"` // the sum of two integers
}

func sumFunc(ctx tool.Context, input SumArgs) (SumResult, error) {
	return SumResult{Sum: input.A + input.B}, nil
}

var expectedNonPartialLLMResponse25Flash = []*model.LLMResponse{
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 2.0,
					"b": 3.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 4.0,
					"b": 5.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 6.0,
					"b": 7.0,
				}),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 5.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 9.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 13.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("The sum of 2 and 3 is 5.\nThe sum of 4 and 5 is 9.\nThe sum of 6 and 7 is 13."),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 10.0,
					"b": 20.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 40.0,
					"b": 50.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 60.0,
					"b": 70.0,
				}),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 30.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 90.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 130.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("The sum of 10 and 20 is 30.\nThe sum of 40 and 50 is 90.\nThe sum of 60 and 70 is 130."),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
}

var expectedNonPartialLLMResponse3FlashPreview = []*model.LLMResponse{
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 2.0,
					"b": 3.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 4.0,
					"b": 5.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 6.0,
					"b": 7.0,
				}),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 5.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 9.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 13.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("The sum of 2 and 3 is 5, the sum of 4 and 5 is 9, and the sum of 6 and 7 is 13."),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 10.0,
					"b": 20.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 40.0,
					"b": 50.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 60.0,
					"b": 70.0,
				}),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 30.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 90.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 130.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("The sum of 10 and 20 is 30, the sum of 40 and 50 is 90, and the sum of 60 and 70 is 130."),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
}

var expectedNonPartialLLMResponse3ProPreview = []*model.LLMResponse{
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 2.0,
					"b": 3.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 4.0,
					"b": 5.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 6.0,
					"b": 7.0,
				}),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 5.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 9.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 13.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("Here are the results of your additions:\n* 2 + 3 = 5\n* 4 + 5 = 9\n* 6 + 7 = 13"),
				{}, // empty part with thought signature
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 10.0,
					"b": 20.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 40.0,
					"b": 50.0,
				}),
				genai.NewPartFromFunctionCall("sum", map[string]any{
					"a": 60.0,
					"b": 70.0,
				}),
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 30.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 90.0,
				}),
				genai.NewPartFromFunctionResponse("sum", map[string]any{
					"sum": 130.0,
				}),
			},
			Role: "user",
		},
	},
	{
		Partial: false,
		Content: &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText("Here are the results for those additions:\n* 10 + 20 = 30\n* 40 + 50 = 90\n* 60 + 70 = 130"),
				{}, // empty part with thought signature
			},
			Role: "model",
		},
		FinishReason: genai.FinishReasonStop,
	},
}

func TestParallelFunctionCalls(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		want      *model.LLMResponse
	}{
		{"gemini-2.5-flash", "gemini-2.5-flash", expectedNonPartialLLMResponse25Flash[0]},
		{"gemini-3-flash-preview", "gemini-3-flash-preview", expectedNonPartialLLMResponse3FlashPreview[0]},
		{"gemini-3.1-pro-preview", "gemini-3.1-pro-preview", expectedNonPartialLLMResponse3ProPreview[0]},
	}
	for _, tt := range tests {
		t.Run("test_parallel_function_calls_"+tt.name, func(t *testing.T) {
			httpRecordFilename := filepath.Join("testdata", strings.ReplaceAll(t.Name(), "/", "_")+".httprr")
			geminiModel, err := gemini.NewModel(t.Context(), tt.modelName, testutil.NewGeminiTestClientConfig(t, httpRecordFilename))
			if err != nil {
				t.Fatal(err)
			}

			sumTool, err := functiontool.New(functiontool.Config{
				Name:        "sum",
				Description: "sums two integers",
			}, sumFunc)
			if err != nil {
				t.Fatal(err)
			}
			type declarer interface {
				Declaration() *genai.FunctionDeclaration
			}
			sumToolWithDeclaration, ok := sumTool.(declarer)
			if !ok {
				t.Fatal("sum tool does not expose a GenAI declaration")
			}

			req := &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Parts: []*genai.Part{
							genai.NewPartFromText("Can you add 2 and 3? Also 4 and 5? And 6 and 7?"),
						},
						Role: "user",
					},
				},
				Config: &genai.GenerateContentConfig{
					SystemInstruction: &genai.Content{
						Parts: []*genai.Part{
							genai.NewPartFromText("You are a calculator assistant. You will recieve requests to add two integers. Respond with the sum of the two integers and you must use the sum tool to calculate the sum.\n\nYou are an agent. Your internal name is \"calculator\". The description about you is \"A calculator that can add two integers\"."),
						},
						Role: "user",
					},
					Tools: []*genai.Tool{
						{
							FunctionDeclarations: []*genai.FunctionDeclaration{
								sumToolWithDeclaration.Declaration(),
							},
						},
					},
				},
			}

			it := geminiModel.GenerateContent(t.Context(), req, true)

			functionCallsPartial := make([]*genai.FunctionCall, 0)
			nonPartialResponses := make([]*model.LLMResponse, 0)
			for resp, err := range it {
				if err != nil {
					t.Fatal(err)
				}
				if resp == nil || resp.Content == nil {
					t.Fatal("expected non-nil response content")
				}
				if !resp.Partial {
					nonPartialResponses = append(nonPartialResponses, resp)
				}
				for _, part := range resp.Content.Parts {
					if part.FunctionCall != nil && resp.Partial {
						functionCallsPartial = append(functionCallsPartial, part.FunctionCall)
					}
				}
			}

			ignoreFields := []cmp.Option{
				cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
				cmpopts.IgnoreFields(genai.Part{}, "ThoughtSignature"),
				cmpopts.IgnoreFields(model.LLMResponse{}, "UsageMetadata"),
			}

			if len(functionCallsPartial) != 3 {
				t.Errorf("expected 3 partial function calls, got %d", len(functionCallsPartial))
			}
			if len(nonPartialResponses) != 1 {
				t.Fatalf("expected 1 non-partial response, got %d", len(nonPartialResponses))
			}

			if diff := cmp.Diff(tt.want, nonPartialResponses[0], ignoreFields...); diff != "" {
				t.Errorf("diff in final response (-want +got): %v", diff)
			}
			for i, part := range nonPartialResponses[0].Content.Parts {
				if part.FunctionCall != nil && len(part.ThoughtSignature) == 0 {
					t.Errorf("final response Parts[%d] (%s): expected non-empty thought signature, got empty", i, part.FunctionCall.Name)
				}
			}
		})
	}
}
