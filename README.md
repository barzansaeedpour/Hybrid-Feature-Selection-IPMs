# A Triple-Component Hybrid Framework for Enhanced Feature Selection: Bridging Filter and Wrapper Methods

This is the implementation of our paper:

A Triple-Component Hybrid Framework for Enhanced Feature Selection: Bridging Filter and Wrapper Methods


This paper proposes a novel triple-component hybrid framework in which an interface component is mediating between filter and wrapper to tackle the contradictions between their objectives. In fact, an adaptable layer is utilized to facilitate a synergistic and implicit collaboration between the filter and wrapper components. This interface introduces some models, namely Importance Probability Models (IPMs), to estimate the significance of features as the mapped data between the components. IPMs are initialized using the filter method and interact with the wrapper method to direct the wrapper one toward identifying the most significant features. The IPMs are updated during the wrapper's evolution to benefit the wrapper's searchability, after the proper guiding in the early stages, by gradually diminishing the filter effect on the IPMs. Here, a NSGA-II-based wrapper method benefiting a new IPMs-based mutation operator is introduced to select features for mutation based on their Importance probabilities. This framework promotes exploration in the early stages and exploitation as the search progresses.