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

package openaimodel

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime"
	"sort"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// buildOpenAIParams converts a generic LLMRequest into the OpenAI-specific
// responses.ResponseNewParams format, preparing it for an API call.
func buildOpenAIParams(modelName string, req *model.LLMRequest) (responses.ResponseNewParams, error) {
	if req == nil {
		return responses.ResponseNewParams{}, ErrRequestNil
	}

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(modelName),
	}
	if req.Model != "" {
		params.Model = shared.ResponsesModel(req.Model)
	}

	// We convert the generic content parts into OpenAI's input format.
	input, err := convertContents(req.Contents)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	if len(input) == 0 {
		return responses.ResponseNewParams{}, ErrNoContents
	}
	params.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: input,
	}

	// Apply generation configuration settings like temperature and max output tokens.
	if err := applyGenerationConfig(&params, req.Config); err != nil {
		return responses.ResponseNewParams{}, err
	}

	// Convert any specified tools into the OpenAI tool format.
	tools, err := convertTools(req.Config)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	// Handle tool choice configuration, if provided.
	if cfg := req.Config; cfg != nil && cfg.ToolConfig != nil {
		choice, err := convertToolChoice(cfg.ToolConfig)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
		if choice != nil {
			params.ToolChoice = *choice
		}
	}

	return params, nil
}

func applyStatelessConfig(params *responses.ResponseNewParams) {
	params.Store = param.NewOpt(false)
	for _, include := range params.Include {
		if include == responses.ResponseIncludableReasoningEncryptedContent {
			return
		}
	}
	params.Include = append(params.Include, responses.ResponseIncludableReasoningEncryptedContent)
}

func convertContents(contents []*genai.Content) (responses.ResponseInputParam, error) {
	var (
		items        responses.ResponseInputParam
		tracker      callTracker
		messageParts responses.ResponseInputMessageContentListParam
		reasoning               = make(map[string]struct{})
		curRole      genai.Role = genai.RoleUser
		flushMessage            = func() error {
			if len(messageParts) == 0 {
				return nil
			}
			msg, err := newMessage(curRole, messageParts)
			if err != nil {
				return err
			}
			if msg != nil {
				items = append(items, responses.ResponseInputItemUnionParam{OfMessage: msg})
			}
			messageParts = nil
			return nil
		}
	)

	for _, content := range contents {
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		curRole = genai.Role(content.Role)
		for _, part := range content.Parts {
			if part != nil && curRole == genai.RoleModel {
				reasoningParam, handled := decodeReasoningPart(part)
				if handled {
					if err := flushMessage(); err != nil {
						return nil, err
					}
					signature := string(part.ThoughtSignature)
					if _, seen := reasoning[signature]; reasoningParam != nil && !seen {
						items = append(items, responses.ResponseInputItemUnionParam{OfReasoning: reasoningParam})
						reasoning[signature] = struct{}{}
					}
					if part.Thought {
						continue
					}
				}
			}
			switch {
			case part == nil:
				continue
			case part.Thought:
				continue
			case part.Text != "":
				if strings.TrimSpace(part.Text) != "" {
					messageParts = append(messageParts, responses.ResponseInputContentUnionParam{
						OfInputText: &responses.ResponseInputTextParam{Text: part.Text, Type: constant.InputText("input_text")},
					})
				}
			case part.InlineData != nil:
				inline, err := convertInlineData(part.InlineData)
				if err != nil {
					return nil, err
				}
				messageParts = append(messageParts, inline)
			case part.FunctionCall != nil:
				if err := flushMessage(); err != nil {
					return nil, err
				}
				callParam, err := tracker.newFunctionCall(part.FunctionCall)
				if err != nil {
					return nil, err
				}
				items = append(items, responses.ResponseInputItemUnionParam{OfFunctionCall: callParam})
			case part.FunctionResponse != nil:
				if err := flushMessage(); err != nil {
					return nil, err
				}
				respParam, err := tracker.newFunctionResponse(part.FunctionResponse)
				if err != nil {
					return nil, err
				}
				items = append(items, responses.ResponseInputItemUnionParam{OfFunctionCallOutput: respParam})
			default:
				return nil, fmt.Errorf("openai: unsupported content part %T", part)
			}
		}
		if err := flushMessage(); err != nil {
			return nil, err
		}
	}

	return items, nil
}

func decodeReasoningPart(part *genai.Part) (*responses.ResponseReasoningItemParam, bool) {
	if part == nil || !bytes.HasPrefix(part.ThoughtSignature, []byte(openAIReasoningSignaturePrefix)) {
		return nil, false
	}
	raw := bytes.TrimPrefix(part.ThoughtSignature, []byte(openAIReasoningSignaturePrefix))
	var reasoning responses.ResponseReasoningItemParam
	if err := json.Unmarshal(raw, &reasoning); err != nil {
		return nil, true
	}
	return &reasoning, true
}

func newMessage(role genai.Role, content responses.ResponseInputMessageContentListParam) (*responses.EasyInputMessageParam, error) {
	if len(content) == 0 {
		return nil, nil
	}
	msgRole, err := normalizeRole(role)
	if err != nil {
		return nil, err
	}
	return &responses.EasyInputMessageParam{
		Role: msgRole,
		Type: responses.EasyInputMessageTypeMessage,
		Content: responses.EasyInputMessageContentUnionParam{
			OfInputItemContentList: content,
		},
	}, nil
}

func convertInlineData(blob *genai.Blob) (responses.ResponseInputContentUnionParam, error) {
	mediaType, _, err := mime.ParseMediaType(blob.MIMEType)
	if err != nil {
		return responses.ResponseInputContentUnionParam{}, fmt.Errorf("%w: %s", ErrUnsupportedInlineDataMIMEType, blob.MIMEType)
	}

	dataURL := func() string {
		return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(blob.Data)
	}
	switch mediaType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return responses.ResponseInputContentUnionParam{OfInputImage: &responses.ResponseInputImageParam{
			Detail:   responses.ResponseInputImageDetailAuto,
			ImageURL: param.NewOpt(dataURL()),
			Type:     constant.InputImage("input_image"),
		}}, nil
	case "application/pdf":
		filename := blob.DisplayName
		if filename == "" {
			filename = "document.pdf"
		}
		return responses.ResponseInputContentUnionParam{OfInputFile: &responses.ResponseInputFileParam{
			FileData: param.NewOpt(dataURL()),
			Filename: param.NewOpt(filename),
			Type:     constant.InputFile("input_file"),
		}}, nil
	default:
		if strings.HasPrefix(mediaType, "text/") {
			return responses.ResponseInputContentUnionParam{OfInputText: &responses.ResponseInputTextParam{
				Text: string(blob.Data),
				Type: constant.InputText("input_text"),
			}}, nil
		}
		return responses.ResponseInputContentUnionParam{}, fmt.Errorf("%w: %s", ErrUnsupportedInlineDataMIMEType, blob.MIMEType)
	}
}

func normalizeRole(role genai.Role) (responses.EasyInputMessageRole, error) {
	switch role {
	case "", genai.RoleUser:
		return responses.EasyInputMessageRoleUser, nil
	case genai.RoleModel:
		return responses.EasyInputMessageRoleAssistant, nil
	case "system":
		return responses.EasyInputMessageRoleSystem, nil
	case "developer":
		return responses.EasyInputMessageRoleDeveloper, nil
	default:
		return "", fmt.Errorf("openai: unsupported role %q", role)
	}
}

// callTracker helps us manage function call IDs, ensuring that function responses
// can be correctly associated with their corresponding calls, especially when IDs are not
// explicitly provided in the input.
type callTracker struct {
	nextID  int
	pending []string
}

// newFunctionCall converts a generic genai.FunctionCall into an OpenAI-specific
// ResponseFunctionToolCallParam. We generate a unique callID if one isn't
// provided, and then marshal the function arguments into a JSON string.
func (t *callTracker) newFunctionCall(fc *genai.FunctionCall) (*responses.ResponseFunctionToolCallParam, error) {
	if fc.Name == "" {
		return nil, ErrFunctionCallMissingName
	}
	callID := fc.ID
	if callID == "" {
		callID = fmt.Sprintf("adk-openai-call-%d", t.nextID)
		t.nextID++
	}
	t.pending = append(t.pending, callID)
	argsValue := fc.Args
	if argsValue == nil {
		argsValue = map[string]any{}
	}
	args, err := json.Marshal(argsValue)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal function args: %w", err)
	}
	return &responses.ResponseFunctionToolCallParam{
		Name:      fc.Name,
		CallID:    callID,
		Arguments: string(args),
		Type:      constant.FunctionCall("function_call"),
	}, nil
}

// newFunctionResponse converts a generic genai.FunctionResponse into an OpenAI-specific
// ResponseInputItemFunctionCallOutputParam. We try to match the response to a pending
// function call. If an explicit callID is provided, we find and remove it from our
// pending list. Otherwise, we assume it corresponds to the oldest pending call.
func (t *callTracker) newFunctionResponse(fr *genai.FunctionResponse) (*responses.ResponseInputItemFunctionCallOutputParam, error) {
	callID := fr.ID
	if callID == "" {
		if len(t.pending) == 0 {
			return nil, fmt.Errorf("openai: response for %q missing call id", fr.Name)
		}
		callID = t.pending[0]
		t.pending = t.pending[1:]
	} else {
		found := false
		for i, pending := range t.pending {
			if pending == callID {
				t.pending = append(t.pending[:i], t.pending[i+1:]...)
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("openai: received function response for unknown or already completed call id %q", callID)
		}
	}
	payload, err := json.Marshal(fr.Response)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal function response: %w", err)
	}
	return &responses.ResponseInputItemFunctionCallOutputParam{
		CallID: callID,
		Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfString: param.NewOpt(string(payload)),
		},
		Type: constant.FunctionCallOutput("function_call_output"),
	}, nil
}

// applyGenerationConfig translates our generic generation configuration into
// OpenAI-specific parameters. We also validate and return errors for features
// that are not supported by the OpenAI Responses API.
func applyGenerationConfig(params *responses.ResponseNewParams, cfg *genai.GenerateContentConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}
	if cfg.TopK != nil {
		return ErrTopKNotSupported
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if len(cfg.StopSequences) > 0 {
		return ErrStopSequencesNotSupported
	}
	if cfg.CandidateCount > 1 {
		return ErrMultipleCandidatesNotSupported
	}
	if cfg.FrequencyPenalty != nil || cfg.PresencePenalty != nil {
		return ErrPenaltiesNotSupported
	}
	if cfg.ResponseLogprobs {
		if cfg.Logprobs != nil {
			params.TopLogprobs = param.NewOpt(int64(*cfg.Logprobs))
		} else {
			params.TopLogprobs = param.NewOpt(int64(1))
		}
		// Responses returns logprobs only when explicitly included.
		params.Include = append(params.Include, responses.ResponseIncludableMessageOutputTextLogprobs)
	}
	if cfg.SystemInstruction != nil {
		inst, err := flattenContentText(cfg.SystemInstruction)
		if err != nil {
			return fmt.Errorf("openai: system instruction: %w", err)
		}
		if inst != "" {
			params.Instructions = param.NewOpt(inst)
		}
	}
	if thinking := cfg.ThinkingConfig; thinking != nil {
		effort := strings.ToLower(string(thinking.ThinkingLevel))
		if effort == string(shared.ReasoningEffortMinimal) && strings.HasPrefix(strings.ToLower(string(params.Model)), "gpt-5.6") {
			effort = string(shared.ReasoningEffortLow)
		}
		switch shared.ReasoningEffort(effort) {
		case "", shared.ReasoningEffort("thinking_level_unspecified"):
		case shared.ReasoningEffortNone,
			shared.ReasoningEffortMinimal,
			shared.ReasoningEffortLow,
			shared.ReasoningEffortMedium,
			shared.ReasoningEffortHigh,
			shared.ReasoningEffortXhigh,
			shared.ReasoningEffortMax:
			params.Reasoning.Effort = shared.ReasoningEffort(effort)
		default:
			return fmt.Errorf("%w: %s", ErrThinkingLevelNotSupported, thinking.ThinkingLevel)
		}
		if thinking.IncludeThoughts {
			params.Reasoning.Summary = shared.ReasoningSummaryAuto
		}
	}
	if cfg.ResponseMIMEType != "" && cfg.ResponseMIMEType != "text/plain" && cfg.ResponseMIMEType != "application/json" {
		return fmt.Errorf("%w: %s", ErrUnsupportedMIMEType, cfg.ResponseMIMEType)
	}
	if cfg.ResponseMIMEType == "application/json" || cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil {
		if cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
			obj := shared.NewResponseFormatJSONObjectParam()
			params.Text = responses.ResponseTextConfigParam{
				Format: responses.ResponseFormatTextConfigUnionParam{
					OfJSONObject: &obj,
				},
			}
		} else {
			format, err := newJSONSchemaFormat(cfg)
			if err != nil {
				return err
			}
			params.Text = responses.ResponseTextConfigParam{
				Format: responses.ResponseFormatTextConfigUnionParam{
					OfJSONSchema: format,
				},
			}
		}
	}
	if cfg.Labels != nil {
		return ErrLabelsNotSupported
	}
	return nil
}

func flattenContentText(content *genai.Content) (string, error) {
	if content == nil {
		return "", nil
	}
	var b strings.Builder
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		if part.Text == "" {
			return "", fmt.Errorf("non-text system instruction part %T", part)
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(part.Text)
	}
	return b.String(), nil
}

// newJSONSchemaFormat constructs an OpenAI-specific JSON schema format from our
// generic GenerateContentConfig. We handle cases where the schema is provided
// directly or needs to be converted, and assign a name to it.
func newJSONSchemaFormat(cfg *genai.GenerateContentConfig) (*responses.ResponseFormatTextJSONSchemaConfigParam, error) {
	var (
		schema map[string]any
		err    error
	)
	switch {
	case cfg.ResponseJsonSchema != nil:
		schema, err = normalizeSchema(cfg.ResponseJsonSchema)
	case cfg.ResponseSchema != nil:
		schema, err = schemaToMap(cfg.ResponseSchema)
	default:
		return nil, fmt.Errorf("openai: json schema requested without schema")
	}
	if err != nil {
		return nil, err
	}
	enforceStrictOpenAISchema(schema)
	name := "adk_response"
	if cfg.ResponseSchema != nil && cfg.ResponseSchema.Title != "" {
		name = cfg.ResponseSchema.Title
	}
	return &responses.ResponseFormatTextJSONSchemaConfigParam{
		Name:   name,
		Schema: schema,
		Strict: param.NewOpt(true),
		Type:   constant.JSONSchema("json_schema"),
	}, nil
}

func normalizeSchema(schema any) (map[string]any, error) {
	switch s := schema.(type) {
	case map[string]any:
		return s, nil
	case nil:
		return nil, ErrEmptyJSONSchema
	default:
		bytes, err := json.Marshal(s)
		if err != nil {
			return nil, fmt.Errorf("openai: marshal json schema: %w", err)
		}
		var result map[string]any
		if err := json.Unmarshal(bytes, &result); err != nil {
			return nil, fmt.Errorf("openai: unmarshal json schema: %w", err)
		}
		return result, nil
	}
}

// enforceStrictOpenAISchema recursively walks the schema and enforces the rules
// required by OpenAI's structured outputs with strict=true. Specifically, it
// sets additionalProperties=false on all object types, and ensures that all
// properties are listed in the required array.
func enforceStrictOpenAISchema(val any) {
	schema, ok := val.(map[string]any)
	if !ok {
		return
	}

	if _, hasRef := schema["$ref"]; hasRef {
		for key := range schema {
			if key != "$ref" {
				delete(schema, key)
			}
		}
		return
	}

	t, hasType := schema["type"]
	isObj := hasType && t == "object"
	propsVal, hasProps := schema["properties"]

	if isObj && hasProps {
		schema["additionalProperties"] = false
		if propsMap, ok := propsVal.(map[string]any); ok {
			req := make([]string, 0, len(propsMap))
			for k := range propsMap {
				req = append(req, k)
			}
			sort.Strings(req)
			schema["required"] = req
		}
	}

	if defsVal, ok := schema["$defs"]; ok {
		if defsMap, ok := defsVal.(map[string]any); ok {
			for _, defn := range defsMap {
				enforceStrictOpenAISchema(defn)
			}
		}
	}

	if hasProps {
		if propsMap, ok := propsVal.(map[string]any); ok {
			for _, prop := range propsMap {
				enforceStrictOpenAISchema(prop)
			}
		}
	}

	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if arrVal, ok := schema[key]; ok {
			if arr, ok := arrVal.([]any); ok {
				for _, item := range arr {
					enforceStrictOpenAISchema(item)
				}
			}
		}
	}

	if itemsVal, ok := schema["items"]; ok {
		if _, isMap := itemsVal.(map[string]any); isMap {
			enforceStrictOpenAISchema(itemsVal)
		}
	}
}
