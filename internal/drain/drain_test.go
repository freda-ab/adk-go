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

import (
	"context"
	"testing"
)

func TestFromContext_NoValue(t *testing.T) {
	if ch := FromContext(context.Background()); ch != nil {
		t.Errorf("FromContext on plain context = %v, want nil", ch)
	}
}

func TestRoundTrip(t *testing.T) {
	ch := make(chan struct{})
	ctx := ToContext(context.Background(), ch)
	got := FromContext(ctx)
	if got != ch {
		t.Error("FromContext did not return the same channel stored by ToContext")
	}
}

func TestSignaled_Nil(t *testing.T) {
	if Signaled(nil) {
		t.Error("Signaled(nil) = true, want false")
	}
}

func TestSignaled_Open(t *testing.T) {
	ch := make(chan struct{})
	if Signaled(ch) {
		t.Error("Signaled(open channel) = true, want false")
	}
}

func TestSignaled_Closed(t *testing.T) {
	ch := make(chan struct{})
	close(ch)
	if !Signaled(ch) {
		t.Error("Signaled(closed channel) = false, want true")
	}
}
