# Abstract
- transition system for dependency
    - arcs only between adjacent words
    - can parse non-projective trees by swapping words
- because of swapping, linear -> quadratic
    - but turns out to still be linear in corpora

# Introduction
- transition-based parsers usually run in linear/quad
    - greedy deterministic search
    - or fixed-width beam search
- but discontinuous syntactic constructions are not solved satisfactorily
    - modelled by non-projective trees
    - which makes parsing computationally harder (NP-hard)
    - also has an impact on accuracy
    - but can't be ignored
- current approaches use one of these strategies
    - non-standard parsing algorithm (combines non-adj substructures)
    - recover non-proj dependencies from output of a projective parser
    - here we will propose algo that derives non-proj trees by reordering ip

# Background Notions
## Dependency Graphs and Trees
- L is a set of dependency labels
- x is a sentence, words w1, ..., wn
- a dependency graph for x is G = (Vx, A)
    - Vx = [0..n] is a set of nodes
      - 0 is extra artificial root node
    - A \subseteq Vx \times L \times Vx is a set of labelled arcs
- for well-formedness, root is at 0

## Transition Systems
- a transition system is a quadruple S = (C, T, cs, Ct)
    - C is a set of configurations
      - here config is split of x, along with set of arcs
    - T is a set of transitions: C -> C
    - cs is an initialisation fn: sentence -> C
      - cs = \x -> ([0], [1..n], {})
    - Ct \subseteq C is a set of terminal configs
      - Ct = ([0], [], _ )

- in a config, x is split into Σ and B
    - Σ = stack, B = buffer
    - c = ([σ|i], [j|b], A)
      - stack with head i
      - buffer with head b
- a transition sequence for x is C{0,m} = (c0,...,cm)
    - c0 = cs(x)
    - cm \in Ct
    - ci = t(c{i-1}) for some t \in T
- the parse assigned to S by C{0,m}
    - G{cm} = (Vx,A{cm})
    - _ , _ , A{cm} = cm

- S is sound for a class Γ of dependencey graphs iff
    - \forall x, C{0,m}
      G{cm} \in Γ
- S is complete for Γ iff
    - \forall x, actual G
      C{0,m} \in S s.t. G{cm} = G

## 2.3 Deterministic Transition-Based Parsing
- an oracle for a transition system S is a fn o : C -> T
```hs
parse :: Oracle -> Sentence -> Graph
parse o x = let initConf = cs x
                lastConf = until (\c -> not $ c `elem` finalConfs)
                                 (\c -> let transn = o c
                                        in transn c)
                                 (initConf)
                (_, _, arcs) = lastConf
            in ([0..(length x)], arcs)
```
- exactly one sequence for a sentence
- if oracle and transition are constant, worst case bound is given by length
- in practical systems, oracle is a classifer trained on treebank data

# Transitions for Dependency Parsing
- now we'll define T

## Projective Dependency Parsing
- minimal transition set Tp
    - LEFTARC{l}
      ([σ|i,j] , B , A) ->
      ([σ|  j] , B , A+(i,l,j))
      - allowed if i ≠ 0
    - RIGHTARC{l}
      ([σ|i,j] , B , A) ->
      ([σ|i  ] , B , A+(i,l,j))
    - SHIFT
      ( σ    , [i|β] , A)
      ([σ|i] ,    β  , Α)

- the system Sp = (C, Tp, cs, Ct) is sound and complete
- number of transitions is exactly 2n

## Unrestricted Dependency Parsing
- extended transition set Tu
    - SWAP
      ([σ|i,j] ,    β  , A)
      ([σ|  j] , [i|β] , A)
      - allowed if 0 < i < j and i ≠ 0
- Su = (C, Tu, cs, Ct) is sound and complete
    - for set including non-proj trees
    - proof is there

- oracle given target arcset A
    - LEFTARC{l} if
      - c = ([σ|i,j], B, Ac)
      - (j,l,i) \in A
      - Ai \subset Ac
    - RIGHTARC{l} if
      - c = ([σ|i,j], B, Ac)
      - (i,l,j) \in A
      - Aj \subseteq Ac
    - SWAP
      - c = ([σ|i,j], B, Ac)
      - j <G i
    - SHIFT o/w

- for time complexity, 2n is lower bound
    - O(n^2) in worst case (n+n^2)

- average case seems to be linear

# Experiments
## Running Time
- we'll check number of transitions
- first on training set
- then on testset using trained oracle

- not much more than 2n

## Parsing Accuracy
- not bad

# Related Work
- recently proposed also
    - but cannot handle unrestricted trees
    - sound but not complete

# Conclusion
