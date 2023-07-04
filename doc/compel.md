# Table of Contents

* [compel.compel](#compel.compel)
  * [Compel](#compel.compel.Compel)
    * [\_\_init\_\_](#compel.compel.Compel.__init__)
    * [make\_conditioning\_scheduler](#compel.compel.Compel.make_conditioning_scheduler)
    * [build\_conditioning\_tensor](#compel.compel.Compel.build_conditioning_tensor)
    * [\_\_call\_\_](#compel.compel.Compel.__call__)
    * [parse\_prompt\_string](#compel.compel.Compel.parse_prompt_string)
    * [describe\_tokenization](#compel.compel.Compel.describe_tokenization)
    * [build\_conditioning\_tensor\_for\_conjunction](#compel.compel.Compel.build_conditioning_tensor_for_conjunction)
    * [build\_conditioning\_tensor\_for\_prompt\_object](#compel.compel.Compel.build_conditioning_tensor_for_prompt_object)
    * [pad\_conditioning\_tensors\_to\_same\_length](#compel.compel.Compel.pad_conditioning_tensors_to_same_length)

<a id="compel.compel"></a>

# compel.compel

<a id="compel.compel.Compel"></a>

## Compel Objects

```python
class Compel()
```

<a id="compel.compel.Compel.__init__"></a>

#### \_\_init\_\_

```python
def __init__(tokenizer: CLIPTokenizer,
             text_encoder: CLIPTextModel,
             textual_inversion_manager: Optional[
                 BaseTextualInversionManager] = None,
             dtype_for_device_getter: Callable[
                 [torch.device], torch.dtype] = lambda device: torch.float32,
             truncate_long_prompts: bool = True,
             padding_attention_mask_value: int = 1,
             downweight_mode: DownweightMode = DownweightMode.MASK,
             use_penultimate_clip_layer: bool = False,
             device: Optional[str] = None)
```

Initialize Compel. The tokenizer and text_encoder can be lifted directly from any DiffusionPipeline.

`textual_inversion_manager`: Optional instance to handle expanding multi-vector textual inversion tokens.
`dtype_for_device_getter`: A Callable that returns a torch dtype for a given device. You probably don't need to
    use this.
`truncate_long_prompts`: if True, truncate input prompts to 77 tokens long including beginning/end markers
    (default behaviour).
    If False, do not truncate, and instead assemble as many 77 token long chunks, each capped by beginning/end
    markers, as is necessary to encode the whole prompt. You will likely need to supply both positive and
    negative prompts in this case - use `pad_conditioning_tensors_to_same_length` to prevent having tensor
    length mismatch errors when passing the embeds on to your DiffusionPipeline for inference.
`padding_attention_mask_value`: Value to write into the attention mask for padding tokens. Stable Diffusion needs 1.
`downweight_mode`: Specifies whether downweighting should be applied by MASKing out the downweighted tokens
    (default) or REMOVEing them (legacy behaviour; messes up position embeddings of tokens following).
`use_penultimate_clip_layer`: If True, use the penultimate hidden layer output of the CLIP text encoder's output,
    rather than the final hidden layer output.
`device`: The torch device on which the tensors should be created. If a device is not specified, the device will
    be the same as that of the `text_encoder` at the moment when `build_conditioning_tensor()` is called.

<a id="compel.compel.Compel.make_conditioning_scheduler"></a>

#### make\_conditioning\_scheduler

```python
def make_conditioning_scheduler(
        positive_prompt: str,
        negative_prompt: str = '') -> ConditioningScheduler
```

Return a ConditioningScheduler object that provides conditioning tensors for different diffusion steps (currently
not fully implemented).

<a id="compel.compel.Compel.build_conditioning_tensor"></a>

#### build\_conditioning\_tensor

```python
def build_conditioning_tensor(text: str) -> torch.Tensor
```

Build a conditioning tensor by parsing the text for Compel syntax, constructing a Conjunction, and then
building a conditioning tensor from that Conjunction.

<a id="compel.compel.Compel.__call__"></a>

#### \_\_call\_\_

```python
@torch.no_grad()
def __call__(text: Union[str, List[str]]) -> torch.FloatTensor
```

Take a string or a list of strings and build conditioning tensors to match.

If multiple strings are passed, the resulting tensors will be padded until they have the same length.

**Returns**:

A tensor consisting of conditioning tensors for each of the passed-in strings, concatenated along dim 0.

<a id="compel.compel.Compel.parse_prompt_string"></a>

#### parse\_prompt\_string

```python
@classmethod
def parse_prompt_string(cls, prompt_string: str) -> Conjunction
```

Parse the given prompt string and return a structured Conjunction object that represents the prompt it contains.

<a id="compel.compel.Compel.describe_tokenization"></a>

#### describe\_tokenization

```python
def describe_tokenization(text: str) -> List[str]
```

For the given text, return a list of strings showing how it will be tokenized.

**Arguments**:

- `text`: The text that is to be tokenized.

**Returns**:

A list of strings representing the output of the tokenizer. It's expected that the output list may be
longer than the number of words in `text` because the tokenizer may split words to multiple tokens. Because of
this, word boundaries are indicated in the output with `</w>` strings.

<a id="compel.compel.Compel.build_conditioning_tensor_for_conjunction"></a>

#### build\_conditioning\_tensor\_for\_conjunction

```python
def build_conditioning_tensor_for_conjunction(
        conjunction: Conjunction) -> Tuple[torch.Tensor, dict]
```

Build a conditioning tensor for the given Conjunction object.

**Returns**:

A tuple of (conditioning tensor, options dict). The contents of the options dict depends on the prompt,
at the moment it is only used for returning cross-attention control conditioning data (`.swap()`).

<a id="compel.compel.Compel.build_conditioning_tensor_for_prompt_object"></a>

#### build\_conditioning\_tensor\_for\_prompt\_object

```python
def build_conditioning_tensor_for_prompt_object(
        prompt: Union[Blend, FlattenedPrompt]) -> Tuple[torch.Tensor, dict]
```

Build a conditioning tensor for the given prompt object (either a Blend or a FlattenedPrompt).

<a id="compel.compel.Compel.pad_conditioning_tensors_to_same_length"></a>

#### pad\_conditioning\_tensors\_to\_same\_length

```python
def pad_conditioning_tensors_to_same_length(
        conditionings: List[torch.Tensor]) -> List[torch.Tensor]
```

If `truncate_long_prompts` was set to False on initialization, or if your prompt includes a `.and()` operator,
conditioning tensors do not have a fixed length. This is a problem when using a negative and a positive prompt
to condition the diffusion process. This function pads any of the passed-in tensors, as necessary, to ensure
they all have the same length, returning the padded tensors in the same order they are passed.

**Example**:

    ``` python
    embeds = compel('("a cat playing in the forest", "an impressionist oil painting").and()')
    negative_embeds = compel("ugly, deformed, distorted")
    [embeds, negative_embeds] = compel.pad_conditioning_tensors_to_same_length([embeds, negative_embeds])
    ```

