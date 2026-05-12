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

package drain

import "context"

type ctxKey int

const drainCtxKey ctxKey = 0

func ToContext(ctx context.Context, ch <-chan struct{}) context.Context {
	return context.WithValue(ctx, drainCtxKey, ch)
}

func FromContext(ctx context.Context) <-chan struct{} {
	ch, _ := ctx.Value(drainCtxKey).(<-chan struct{})
	return ch
}

// Signaled does a non-blocking check on whether ch has been closed or sent on.
// Returns false if ch is nil.
func Signaled(ch <-chan struct{}) bool {
	if ch == nil {
		return false
	}
	select {
	case <-ch:
		return true
	default:
		return false
	}
}
