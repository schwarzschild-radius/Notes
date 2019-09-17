# CMU - 15210: Parallel and Sequential Data Structures
## by Guy Blelloch

## Basics - Solving Recurrences
There are three methods to solve recurrences
1. `Tree Method` - Sum up the cost of each node at each level
2. `Brick Method` - If the cost at each level is a multiplicative factor of the previous level. This is direct method to solve recurrences without solving it.
  1. `Balanced` - Cost(L) of every level(d) is roughly equal. Cost is `O(dL)`
  2. `Root Dominated` - Cost(L) of each level is constant factor smaller than the previous level. Cost is `O(L)`
  3. `Leaf Dominated` - Const(L) of each level is constant factor larger than the previous level. Cost is `O(L sub d)`.


## Exercises
