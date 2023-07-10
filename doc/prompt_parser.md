# Table of Contents

* [compel.prompt\_parser](#compel.prompt_parser)
  * [Conjunction](#compel.prompt_parser.Conjunction)
  * [Prompt](#compel.prompt_parser.Prompt)
  * [FlattenedPrompt](#compel.prompt_parser.FlattenedPrompt)
  * [Fragment](#compel.prompt_parser.Fragment)
  * [Attention](#compel.prompt_parser.Attention)
  * [CrossAttentionControlSubstitute](#compel.prompt_parser.CrossAttentionControlSubstitute)
  * [Blend](#compel.prompt_parser.Blend)
  * [PromptParser](#compel.prompt_parser.PromptParser)
    * [parse\_conjunction](#compel.prompt_parser.PromptParser.parse_conjunction)
    * [flatten](#compel.prompt_parser.PromptParser.flatten)

<a id="compel.prompt_parser"></a>

# compel.prompt\_parser

<a id="compel.prompt_parser.Conjunction"></a>

## Conjunction Objects

```python
class Conjunction()
```

Storage for one or more Prompts or Blends, each of which is to be separately diffused and then the results merged
by weighted sum in latent space.

<a id="compel.prompt_parser.Prompt"></a>

## Prompt Objects

```python
class Prompt()
```

Mid-level structure for storing the tree-like result of parsing a prompt. A Prompt may not represent the whole of
the singular user-defined "prompt string" (although it can) - for example, if the user specifies a Blend, the objects
that are to be blended together are stored individuall as Prompt objects.

Nesting makes this object not suitable for directly tokenizing; instead call flatten() on the containing Conjunction
to produce a FlattenedPrompt.

<a id="compel.prompt_parser.FlattenedPrompt"></a>

## FlattenedPrompt Objects

```python
class FlattenedPrompt()
```

A Prompt that has been passed through flatten(). Its children can be readily tokenized.

<a id="compel.prompt_parser.Fragment"></a>

## Fragment Objects

```python
class Fragment(BaseFragment)
```

A Fragment is a chunk of plain text and an optional weight. The text should be passed as-is to the CLIP tokenizer.

<a id="compel.prompt_parser.Attention"></a>

## Attention Objects

```python
class Attention()
```

Nestable weight control for fragments. Each object in the children array may in turn be an Attention object;
weights should be considered to accumulate as the tree is traversed to deeper levels of nesting.

Do not traverse directly; instead obtain a FlattenedPrompt by calling Flatten() on a top-level Conjunction object.

<a id="compel.prompt_parser.CrossAttentionControlSubstitute"></a>

## CrossAttentionControlSubstitute Objects

```python
class CrossAttentionControlSubstitute(CrossAttentionControlledFragment)
```

A Cross-Attention Controlled ('prompt2prompt') fragment, for use inside a Prompt, Attention, or FlattenedPrompt.
Representing an "original" word sequence that supplies feature vectors for an initial diffusion operation, and an
"edited" word sequence, to which the attention maps produced by the "original" word sequence are applied. Intuitively,
the result should be an "edited" image that looks like the "original" image with concepts swapped.

eg "a cat sitting on a car" (original) -> "a smiling dog sitting on a car" (edited): the edited image should look
almost exactly the same as the original, but with a smiling dog rendered in place of the cat. The
CrossAttentionControlSubstitute object representing this swap may be confined to the tokens being swapped:
    CrossAttentionControlSubstitute(original=[Fragment('cat')], edited=[Fragment('dog')])
or it may represent a larger portion of the token sequence:
    CrossAttentionControlSubstitute(original=[Fragment('a cat sitting on a car')],
                                    edited=[Fragment('a smiling dog sitting on a car')])

In either case expect it to be embedded in a Prompt or FlattenedPrompt:
FlattenedPrompt([
        Fragment('a'),
        CrossAttentionControlSubstitute(original=[Fragment('cat')], edited=[Fragment('dog')]),
        Fragment('sitting on a car')
    ])

<a id="compel.prompt_parser.Blend"></a>

## Blend Objects

```python
class Blend()
```

Stores a Blend of multiple Prompts. To apply, build feature vectors for each of the child Prompts and then perform a
weighted blend of the feature vectors to produce a single feature vector that is effectively a lerp between the
Prompts.

<a id="compel.prompt_parser.PromptParser"></a>

## PromptParser Objects

```python
class PromptParser()
```

<a id="compel.prompt_parser.PromptParser.parse_conjunction"></a>

#### parse\_conjunction

```python
def parse_conjunction(prompt: str, verbose: bool = False) -> Conjunction
```

**Arguments**:

- `prompt`: The prompt string to parse

**Returns**:

a Conjunction representing the parsed results.

<a id="compel.prompt_parser.PromptParser.flatten"></a>

#### flatten

```python
def flatten(root: Conjunction, verbose=False) -> Conjunction
```

Flattening a Conjunction traverses all of the nested tree-like structures in each of its Prompts or Blends,

producing from each of these walks a linear sequence of Fragment or CrossAttentionControlSubstitute objects
that can be readily tokenized without the need to walk a complex tree structure.

**Arguments**:

- `root`: The Conjunction to flatten.

**Returns**:

A Conjunction containing the result of flattening each of the prompts in the passed-in root.

