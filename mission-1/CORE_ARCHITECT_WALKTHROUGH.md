# Core Architect — Step-by-Step Walkthrough

## Step 1: Understand the entry point
Open `04_dialogic_engine.py`. `main()` parses command-line arguments, creates an `LLMClient`, runs the selected protocol, and prints/saves a transcript.

## Step 2: Understand provider integration
`LLMClient.complete()` selects OpenAI, Gemini, Anthropic, or offline mock mode. Each live provider method sends an HTTP request using Python's standard library. The API key is read from an environment variable and should never be committed to GitHub.

## Step 3: Understand Socratic discovery
`run_socratic()` maintains the current claim and cycle count. Each model response is parsed as JSON and normalized. The loop stops on `resolved`, `aporia`, an invalid refinement, or the configured cycle limit.

## Step 4: Understand the debate
`run_debate()` cycles through Salviati, Simplicio, and Sagredo. Each persona receives the scenario and recent dialogue. Sagredo then synthesizes the complete trace and a safety audit is appended.

## Step 5: Reproduce the demo
Run `python 04_dialogic_engine.py --offline --mode both`. This does not call an external API and is useful for a classroom demo when Wi-Fi or API access is unavailable.

## Step 6: Test and then run live
Run `python -m unittest discover -s tests -v`. After the offline run succeeds, configure a provider API key and run live mode. Do not claim that the included offline transcript is a live model evaluation.

## Step 7: Explain limitations
The LLM's assessment of contradiction is not a formal proof. Personas are prompted roles, not independent experts. Debate may reveal blind spots but cannot guarantee correctness or eliminate bias. Preserve the transcript and review claims against external evidence.

## Suggested oral-demo script
1. State the initial claim: “An AI agent is benevolent if it obeys human commands.”
2. Show the counterexample involving unauthorized disclosure of private information.
3. Explain why obedience alone is not enough and why the engine refines the claim.
4. Show the aporia state when the meaning of legitimate constraints remains unsettled.
5. Run the Atlas open-source debate and identify one benefit, one risk, and one unresolved question.
6. Finish by explaining that the system supports structured reasoning; it does not certify truth.
