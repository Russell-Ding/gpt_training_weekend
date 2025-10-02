# Weekend GPT Training Assignment: Build and Train Your Own Language Model

**Duration**: 6 hours (Saturday & Sunday, 3 hours each day)  
**Goal**: Build a complete GPT training pipeline optimized for Apple Silicon MPS  
**Difficulty**: Advanced (bring the challenge on! 🔥)

---

## 🎯 **Assignment Overview**

You'll build a production-ready GPT training system from scratch, implementing modern techniques like AdamW optimization, SGDR scheduling, MPS-specific optimizations, and comprehensive W&B monitoring. Each step includes testable components perfect for Claude Code assistance.

---

## 📅 **Day 1: Architecture & Core Pipeline (Saturday - 3 hours)**

### **Step 1: Project Setup & Configuration (30 minutes)**

**Objective**: Create a professional project structure with modern configuration management.

```bash
# Create project structure
mkdir gpt_training_weekend && cd gpt_training_weekend
mkdir -p {src/{models,training,data,evaluation},configs/{model,training},tests/{unit,integration},scripts,notebooks}
```

**Tasks**:
1. Set up `pyproject.toml` with dependencies:
   ```toml
   [project]
   name = "gpt-training-weekend"
   version = "0.1.0"
   dependencies = [
       "torch>=2.0.0",
       "tiktoken",
       "wandb",
       "hydra-core",
       "omegaconf",
       "pytest",
       "numpy",
       "tqdm",
       "psutil"
   ]
   ```

2. Create Hydra configs in `configs/`:
   - `config.yaml` (main config)
   - `model/small_gpt.yaml` (model architecture)
   - `training/mps_optimized.yaml` (training settings)

**Claude Code Task**: 
```bash
# Generate comprehensive pytest configuration
claude code "Create a pytest.ini configuration file and a conftest.py file with common fixtures for PyTorch model testing, including device setup, random seed fixing, and memory cleanup utilities"
```

**Deliverable**: Fully configured project with working imports and test setup.

---

### **Step 2: GPT Model Architecture (45 minutes)**

**Objective**: Implement a modern GPT architecture with MPS optimizations.

**File**: `src/models/gpt.py`

**Tasks**:
1. Implement `GPTConfig` dataclass with MPS-specific parameters
2. Create `TransformerBlock` with:
   - Multi-head attention with chunking support
   - Feed-forward network with GELU activation
   - Layer normalization (NOT RMSNorm yet - that's Week 3!)
   - Residual connections with proper scaling

3. Implement main `SmallGPT` class with:
   - Weight tying between embeddings and output layer
   - Gradient checkpointing support
   - Proper weight initialization (GPT-2 style)

**Key Implementation Details**:
```python
@dataclass
class GPTConfig:
    # Core architecture
    vocab_size: int = 16384
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    block_size: int = 512
    bias: bool = False
    
    # MPS optimizations
    use_gradient_checkpointing: bool = True
    attention_chunk_size: int = 2048
    max_batch_size: int = 8
```

**Claude Code Tasks**:
```bash
# Generate attention mechanism tests
claude code "Create comprehensive unit tests for a multi-head attention layer, testing forward pass shapes, gradient flow, memory efficiency, and edge cases like empty sequences and single tokens"

# Generate model architecture validation
claude code "Write pytest tests that validate GPT model architecture including parameter counting, layer connectivity, weight initialization statistics, and output shape verification for different configurations"
```

**Deliverable**: Working GPT model that passes all architectural tests.

---

### **Step 3: MPS Optimization Layer (45 minutes)**

**Objective**: Implement Apple Silicon specific optimizations.

**File**: `src/training/mps_optimizer.py`

**Tasks**:
1. Create `MPSDeviceManager`:
   - Auto-detect optimal batch size
   - Memory monitoring and cleanup
   - Fallback to CPU when needed

2. Implement `MPSAttentionOptimizer`:
   - Chunked attention for long sequences
   - Buffer size validation
   - Memory-efficient operations

3. Add `MPSMemoryProfiler`:
   - Real-time memory tracking
   - OOM prediction and prevention
   - Performance benchmarking

**Key Features**:
```python
class MPSDeviceManager:
    def __init__(self, target_memory_ratio=0.7):
        self.device = self._setup_device()
        self.max_memory_gb = self._get_available_memory() * target_memory_ratio
        
    def find_optimal_batch_size(self, model, sample_input):
        """Binary search for maximum stable batch size"""
        # Implementation with progressive testing
        
    def safe_forward_pass(self, model, inputs, max_retries=3):
        """Forward pass with automatic OOM recovery"""
        # Implementation with chunking fallback
```

**Claude Code Tasks**:
```bash
# Generate MPS-specific tests
claude code "Create pytest tests for MPS device management including batch size optimization, memory monitoring, OOM recovery, and performance benchmarking across different model sizes"

# Generate memory profiling utilities
claude code "Write a comprehensive memory profiling test suite that tracks PyTorch MPS memory usage patterns, validates cleanup procedures, and tests memory leak detection"
```

**Deliverable**: Robust MPS optimization layer with comprehensive testing.

---

### **Step 4: AdamW + SGDR Training Pipeline (50 minutes)**

**Objective**: Implement modern optimization with proper learning rate scheduling.

**File**: `src/training/trainer.py`

**Tasks**:
1. Create `ModernOptimizer` class:
   - AdamW with β2=0.95 (not 0.999!)
   - Proper weight decay grouping
   - Parameter group management

2. Implement `SGDRScheduler`:
   - Cosine annealing with warm restarts
   - Linear warmup phase
   - Dynamic restart periods

3. Build `MPSMixedPrecisionTrainer`:
   - Manual loss scaling (no GradScaler)
   - Float16 autocast
   - Gradient overflow detection

**Key Implementation**:
```python
class ModernOptimizer:
    @staticmethod
    def configure_adamw(model, lr=6e-4, weight_decay=0.1):
        # Separate decay/no-decay parameters
        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
        
        return torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=lr, betas=(0.9, 0.95), eps=1e-8)

def get_cosine_schedule_with_warmup(step, max_lr=6e-4, min_lr_ratio=0.1, 
                                   warmup_steps=2000, max_steps=100000):
    # Implementation with proper cosine decay
```

**Claude Code Tasks**:
```bash
# Generate optimizer tests
claude code "Create unit tests for AdamW optimizer configuration including parameter grouping validation, learning rate scheduling accuracy, and gradient scaling behavior verification"

# Generate training loop tests
claude code "Write integration tests for the training pipeline including optimizer state management, checkpoint saving/loading, gradient accumulation, and mixed precision training validation"
```

**Deliverable**: Complete training pipeline with modern optimization techniques.

---

## 📅 **Day 2: Data, Training & Evaluation (Sunday - 3 hours)**

### **Step 5: Data Pipeline & Tokenization (45 minutes)**

**Objective**: Build efficient data loading optimized for language model training.

**File**: `src/data/dataset.py`

**Tasks**:
1. Implement `MemoryMappedDataset`:
   - Efficient large file handling
   - Sliding window tokenization
   - Dynamic sequence packing

2. Create `GPTTokenizer`:
   - TikToken integration
   - Vocabulary optimization for small models
   - Batch encoding with attention masks

3. Build `MPS_DataLoader`:
   - No multiprocessing (MPS limitation)
   - Optimal prefetching
   - Memory-conscious batching

**Implementation Focus**:
```python
class MemoryMappedDataset:
    def __init__(self, data_path, block_size, stride=None):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.stride = stride or block_size
        
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        chunk = self.data[start_idx:start_idx + self.block_size + 1]
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])
```

**Claude Code Tasks**:
```bash
# Generate data pipeline tests
claude code "Create comprehensive tests for memory-mapped dataset including boundary conditions, sequence packing efficiency, tokenization accuracy, and batch generation consistency"

# Generate tokenizer validation
claude code "Write tests for GPT tokenizer including vocabulary size validation, encoding/decoding roundtrip accuracy, special token handling, and batch processing correctness"
```

**Deliverable**: Efficient data pipeline ready for large-scale training.

---

### **Step 6: Weights & Biases Integration (30 minutes)**

**Objective**: Implement comprehensive experiment tracking and monitoring.

**File**: `src/training/monitoring.py`

**Tasks**:
1. Create `WandBLogger`:
   - Automatic metric aggregation
   - Model artifact management
   - Real-time visualization

2. Implement `MetricsCollector`:
   - Training loss tracking
   - Learning rate monitoring
   - Memory usage profiling
   - Gradient statistics

3. Build `ExperimentManager`:
   - Configuration logging
   - Checkpoint versioning
   - Automated reporting

**Key Features**:
```python
class WandBLogger:
    def __init__(self, project_name, config):
        wandb.init(project=project_name, config=config)
        self.step = 0
        
    def log_training_step(self, loss, lr, grad_norm, memory_usage):
        wandb.log({
            "train/loss": loss,
            "train/learning_rate": lr,
            "train/grad_norm": grad_norm,
            "system/memory_gb": memory_usage,
            "step": self.step
        })
        self.step += 1
```

**Claude Code Tasks**:
```bash
# Generate monitoring tests
claude code "Create tests for W&B integration including metric logging accuracy, artifact management, configuration tracking, and offline mode fallback functionality"
```

**Deliverable**: Professional experiment tracking with rich visualizations.

---

### **Step 7: Model Evaluation Framework (30 minutes)**

**Objective**: Implement comprehensive model evaluation metrics.

**File**: `src/evaluation/metrics.py`

**Tasks**:
1. Create `PerplexityCalculator`:
   - Efficient batch computation
   - Cross-entropy loss aggregation
   - Statistical significance testing

2. Implement `GenerationEvaluator`:
   - Text quality metrics
   - Diversity measurements
   - Coherence scoring

3. Build `BenchmarkSuite`:
   - Standard evaluation tasks
   - Automated reporting
   - Performance comparison

**Implementation**:
```python
class ModelEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def calculate_perplexity(self, test_data):
        total_loss = 0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_data:
                logits, loss = self.model(batch['input_ids'], batch['targets'])
                total_loss += loss.item() * batch['input_ids'].numel()
                total_tokens += batch['input_ids'].numel()
                
        return torch.exp(torch.tensor(total_loss / total_tokens))
```

**Claude Code Tasks**:
```bash
# Generate evaluation tests
claude code "Create unit tests for model evaluation including perplexity calculation accuracy, generation quality metrics validation, and benchmark consistency across different model checkpoints"
```

**Deliverable**: Comprehensive evaluation framework with standardized metrics.

---

### **Step 8: Complete Training Script (45 minutes)**

**Objective**: Integrate all components into a production-ready training script.

**File**: `scripts/train.py`

**Tasks**:
1. Implement `HydraTrainingScript`:
   - Configuration management
   - Component initialization
   - Error handling and recovery

2. Create `TrainingLoop`:
   - Step-by-step execution
   - Checkpoint management
   - Evaluation scheduling

3. Build `ExperimentRunner`:
   - Multi-run support
   - Hyperparameter sweeps
   - Results aggregation

**Main Training Loop**:
```python
@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # Setup
    wandb_logger = WandBLogger(cfg.experiment_name, cfg)
    model = instantiate_model(cfg.model)
    dataset = create_dataset(cfg.data)
    trainer = MPSMixedPrecisionTrainer(model)
    
    # Training loop
    for step in range(cfg.training.max_steps):
        # Get batch
        batch = dataset.get_batch(cfg.training.batch_size)
        
        # Training step
        loss = trainer.training_step(batch, optimizer)
        
        # Logging and evaluation
        if step % cfg.training.log_interval == 0:
            wandb_logger.log_training_step(loss, lr, grad_norm, memory_usage)
            
        if step % cfg.training.eval_interval == 0:
            eval_metrics = evaluate_model(model, dataset.get_validation_data())
            save_checkpoint(model, optimizer, step, eval_metrics)
    
    # Final evaluation and model card generation
    generate_final_report(model, cfg, eval_metrics)
```

**Claude Code Tasks**:
```bash
# Generate integration tests
claude code "Create end-to-end integration tests for the complete training pipeline including configuration loading, model initialization, training step execution, checkpoint saving, and evaluation reporting"

# Generate performance benchmarks
claude code "Write performance benchmarking tests that measure training speed, memory efficiency, convergence behavior, and compare against expected baselines for different model configurations"
```

**Deliverable**: Complete, tested training system ready for production use.

---

### **Step 9: Training Execution & Validation (30 minutes)**

**Objective**: Run actual training and validate the complete system.

**Tasks**:
1. **Quick Smoke Test** (10 minutes):
   ```bash
   python scripts/train.py training.max_steps=100 training.eval_interval=50
   ```

2. **Small Model Training** (15 minutes):
   ```bash
   python scripts/train.py model=small_gpt training=mps_optimized training.max_steps=1000
   ```

3. **System Validation** (5 minutes):
   - Check W&B dashboard
   - Verify checkpoint saving
   - Test inference generation

**Expected Results**:
- Training loss decreasing
- No MPS memory errors
- Smooth W&B logging
- Successful checkpoint creation
- Basic text generation working

**Claude Code Tasks**:
```bash
# Generate system validation tests
claude code "Create automated system validation tests that verify training convergence, checkpoint integrity, memory stability, and generation quality after training completion"
```

**Deliverable**: Verified working training system with successful training run.

---

## 🎯 **Final Deliverables Checklist**

### **Technical Components**
- [ ] Modern GPT architecture with MPS optimizations
- [ ] AdamW optimizer with proper parameter grouping
- [ ] SGDR learning rate scheduling with cosine annealing
- [ ] Mixed precision training for Apple Silicon
- [ ] Memory-mapped dataset with efficient batching
- [ ] Comprehensive W&B monitoring and logging
- [ ] Model evaluation framework with multiple metrics
- [ ] Production-ready training script with Hydra configuration

### **Quality Assurance**
- [ ] Full unit test suite (95%+ coverage)
- [ ] Integration tests for complete pipeline
- [ ] Performance benchmarks and baselines
- [ ] Memory profiling and optimization validation
- [ ] Error handling and recovery mechanisms

### **Documentation & Results**
- [ ] Training curves and convergence plots
- [ ] Model performance benchmarks
- [ ] Memory usage analysis
- [ ] Generated text samples
- [ ] Model card with specifications and limitations

---

## 🚀 **Bonus Challenges** (If you finish early!)

1. **Multi-GPU Simulation**: Implement gradient accumulation to simulate larger batch sizes
2. **Dynamic Batching**: Adaptive batch size based on sequence length
3. **Custom Tokenizer**: Train a domain-specific BPE tokenizer
4. **Attention Visualization**: Create heatmaps of attention patterns
5. **Model Compression**: Implement knowledge distillation from larger models

---

## 💡 **Claude Code Integration Strategy**

At each step, use Claude Code to:

1. **Generate Test Cases**: 
   ```bash
   claude code "Create comprehensive unit tests for [component] including edge cases and error conditions"
   ```

2. **Validate Implementation**:
   ```bash
   claude code "Review this PyTorch implementation for potential bugs, optimization opportunities, and MPS compatibility issues"
   ```

3. **Performance Optimization**:
   ```bash
   claude code "Analyze this training loop for memory efficiency and suggest MPS-specific optimizations"
   ```

4. **Documentation**:
   ```bash
   claude code "Generate comprehensive docstrings and type hints for this module following Google style guide"
   ```

---

## 🎓 **Learning Outcomes**

By completing this assignment, you'll have:

✅ **Mastered Modern LLM Training**: AdamW, SGDR, mixed precision  
✅ **Apple Silicon Expertise**: MPS optimization and memory management  
✅ **Production Engineering**: Testing, monitoring, configuration management  
✅ **Research Skills**: Evaluation metrics and experiment tracking  
✅ **Portfolio Project**: Complete, documented, testable codebase  

**Ready to build the future of AI on your MacBook Pro? Let's code! 🔥**