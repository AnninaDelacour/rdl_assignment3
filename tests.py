import torch


@torch.no_grad()
def test_lora_layer_forward(lora_class):
    base_layer = torch.nn.Linear(2, 4)
    lora_layer = lora_class(base_layer, rank=3)
    input = torch.randn(1, 2)
    try:
        output = lora_layer(input)
    except Exception as e:
        raise AssertionError(f"Error in forward pass: {e}")
    base_output = base_layer(input)
    assert torch.allclose(output, base_output), "Initial LoRA forward should be identical to base forward, did you initialize the weights correctly?"

def test_lora_layers(model):
    target_modules = ["q_proj", "k_proj", "v_proj", "attn.c_proj"]
    should_be_lora = []
    should_not_be_lora = []
    for name, layer in model.named_modules():
        is_target_module = any(name.endswith(target_module) for target_module in target_modules)
        if is_target_module and not type(layer).__name__ == "LoRALinear":
            should_be_lora.append(name)
        elif not is_target_module and type(layer).__name__ == "LoRALinear":
            should_not_be_lora.append(name)

    assert len(should_be_lora) == 0, f"The following layers should be LoRA layers: {should_be_lora}"
    assert len(should_not_be_lora) == 0, f"The following layers should not be LoRA layers: {should_not_be_lora}"


def test_only_lora_trainable(model, lora_param_names=["lora_A", "lora_B"]):
    should_not_be_trainable = [name for name, param in model.named_parameters() if param.requires_grad and not any(lora_param_name in name for lora_param_name in lora_param_names)]
    assert len(should_not_be_trainable) == 0, f"The following parameters should not be trainable: {should_not_be_trainable}"
