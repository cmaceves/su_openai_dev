import os
import sys
import prompts

def alternate_mechanism(term1, term2):
    """
    Get the mechanism without providing ChatGPT with additional data.
    """
    print(term1, term2)
    all_triples = []
    resp, prompt = prompts.alternate_mechanism(term1, term2)
    steps = resp.split("\n")
    steps = [x.split(". ")[-1] for x in steps]

    #expand the steps to triples
    for step in steps:
        triple = step.split(" -> ")
        all_triples.append(triple)
    print(all_triples)

def main(indication):
    print(indication)

if __name__ == "__main__":
    main()
