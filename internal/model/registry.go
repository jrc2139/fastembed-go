package model

import "sync"

// Registry holds model metadata for a specific model type.
// It is safe for concurrent access.
type Registry[M comparable, I any] struct {
	mu     sync.RWMutex
	models map[M]*I
}

// NewRegistry creates a new registry.
func NewRegistry[M comparable, I any]() *Registry[M, I] {
	return &Registry[M, I]{
		models: make(map[M]*I),
	}
}

// Register adds a model to the registry.
func (r *Registry[M, I]) Register(model M, info *I) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.models[model] = info
}

// RegisterAll adds multiple models to the registry.
func (r *Registry[M, I]) RegisterAll(models map[M]*I) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for m, info := range models {
		r.models[m] = info
	}
}

// Get returns model info or nil if not found.
func (r *Registry[M, I]) Get(model M) *I {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.models[model]
}

// Has checks if a model is registered.
func (r *Registry[M, I]) Has(model M) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.models[model]
	return ok
}

// List returns all registered model infos.
func (r *Registry[M, I]) List() []*I {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]*I, 0, len(r.models))
	for _, info := range r.models {
		result = append(result, info)
	}
	return result
}

// Models returns all registered model identifiers.
func (r *Registry[M, I]) Models() []M {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]M, 0, len(r.models))
	for m := range r.models {
		result = append(result, m)
	}
	return result
}

// Count returns the number of registered models.
func (r *Registry[M, I]) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.models)
}
