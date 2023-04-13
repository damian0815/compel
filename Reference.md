*This document is a stub. You can contribute and help this project by adding more details and examples to this document. Thanks!*

# Prompt Syntax

## Attention Weights
Append a word or phrase with `-` or `+`, or a weight between `0` and `2` (`1` is default), to decrease or increase the 
importance of that word/phrase in the generated image.

You can assign weights to multiple words by using parentheses. For e.g. `a man (picking apricots)1.5` or `a man (picking apricots)++`

You can add more `+` or `-` symbols to increase/decrease the weight further. For e.g. `apricot++` has more importance than `apricot+`, and `apricot--` has less importance than `apricot-`. There's no limit to how many `+` or `-` symbols you can use.

You can also use a number to assign an exact weight to a word/phrase:
- A weight between `0` and `1.0` reduces the importance of the token. For e.g. `(apricots)0.5` reduces the importance of apricots to half.
- A weight between `1.0` and `2` increases the importance of the token. For e.g. `(apricots)1.5` increases the importance of apricots by 1.5 times.

`+` is essentially a weight of `1.1`, and `-` is essentially a weight of `0.9`.

### More examples:
- nesting: `a tall thin man (picking apricots+)++` (`apricots` effectively gets
  `+++`)
- single words without parentheses: `a tall thin man picking apricots+`
- single or multiple words with parentheses:
  `a tall thin man (picking apricots)+` `a tall thin man picking (apricots)-`
  `a tall thin man (picking apricots)-`
- more effect with more symbols: `a tall thin man (picking apricots)++`, and `a tall thin man (picking apricots)+++`
- all of the above with explicit numbers: `a tall thin man picking (apricots)1.1`
  `a tall thin man (picking (apricots)1.3)1.1`. (`+` is equivalent to 1.1, `++`
  is `1.1 x 1.1`, `+++` is `1.1 x 1.1 x 1.1`, etc; `-` means 0.9, `--` means `0.9 x 0.9`,
  etc.)

## Blending between prompts
You can blend between concepts in the prompt by using the `.blend()` function.

For e.g.: `("blue sphere", "red cube").blend(0.25,0.75)`

This will tell the sampler to blend 25% of the concept of a blue sphere with 75%
of the concept of a red cube. The blend weights can use any combination of
integers and floating point numbers.

## Escaping parentheses () and speech marks ""

If the prompt you are using has parentheses `()` or speech marks `""` as part of its
syntax, you will need to "escape" these using a backslash, so that`(my_keyword)`
becomes `\(my_keyword\)`. Otherwise, the prompt parser will attempt to interpret
the parentheses as part of the prompt syntax and it will get confused.
