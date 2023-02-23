**Note: This repo is still in W.I.P**<br>
**And remember. Send complaints to you know who when thing goes wrong.**<br>
**But for now. Don't. It's still in W.I.P!**

## What is this?
This is a repo for a NAS-like algorithm consisting of chromosomes and GA (mainly) for a group project using PyTorch framework.

The goal of these codes are to create a NAS with flexible chromosomes that can be easily understand by everyone.

But if your goal isn't finding the best model. Than you can generate a PyTorch model instantly by using a string and data shape (.shape method).

But since this is still a W.I.P (heavily). Expect changes. Cheerios!

## Requirements:
- torch >= 1.13.1
- automata-lib >= 6.0.2 (v6)

## To-do List:
- [ ] Complete Inception chromosomes
    - [ ] DFA for checking validity (not sure if needed or not). 
    - [ ] Figure out the math for keeping Pooling spatial dimensions.
    - [ ] Chromosome Operators.
      - For CNN parts, just mutate the same
      - For Inception parts, it's either each blocks are independant from each other or the same as first inception block.
    - [ ] Chromosome Processing.
      - Maybe I should try to design the checker that doesn't rely on DFA. Since writing it is pain in the ass.
- [ ] Clean up VGG codes to make other networks in the future independant on its code.
  - [ ] Rewrite the DFA
- [ ] Add a class for creating Linear models.
  - [ ] Chromosome Operators.
  - [ ] Chromosome Processing.
- [x] Complete the support for output channel specify.
  - Used in Inception.