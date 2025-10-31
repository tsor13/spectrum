# Spectrum Repository <img width="50" height="40" alt="spectrum2" src="https://github.com/user-attachments/assets/2af31d90-d60e-4611-9caf-5cbddd1acafa" style="vertical-align: -100px;" />

Companion repository for [*Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability*](https://arxiv.org/abs/2510.06084)

**Paper: https://arxiv.org/abs/2510.06084**

**Models 🤗: https://huggingface.co/collections/tsor13/spectrum-68dac670f618224845c0fb7d**

**Spectrum Suite** <img width="20" height="20" alt="spectrumsuitelogo" src="https://github.com/user-attachments/assets/f5568f9c-84fa-46e1-86cb-ee73fd1c0a84" style="vertical-align: middle;" /> is a large-scale resource compiled from >40 data sources spanning >90 tasks requiring models to steer to and match diverse distributions ranging from varied human preferences to numerical distributions and more. For illustrative examples, please references [this pdf](https://tsor13.github.io/files/spectrumprompts.pdf). To use it for training or evaluation, follow the instructions in this repo.

**Spectrum Tuning** <img width="20" height="20" alt="spectrum2" src="https://github.com/user-attachments/assets/2af31d90-d60e-4611-9caf-5cbddd1acafa" style="vertical-align: middle;" /> is a simple post-training method utilizing Spectrum Suite to teach models to span and steer to distributions described in either natural language, with examples, or both. Training code can be found in this repo, and trained models on huggingface.

## Setup
First, clone the repository. Then,
```
cd spectrum
```

We use uv for package management. To initialize the environment, run
```
uv sync
uv pip install -e .
```

To download the data, run
```
cd data
bash download.sh
cd -
```
(Note: Some data sources require no download and are accessed directly from huggingface, while others require manual download. For reference, see [`data/README.md`](data/README.md))

## In-Context Learning / Steerability Experiments
The main code for running the in-context steerability experiments can be found in [`src/spectrum/icl_classes/eval_icl.py`](src/spectrum/icl_classes/eval_icl.py).

Example commands for replicating the gemma-3-12b experiments on the flight task:
```
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --model_name tsor13/spectrum-gemma-3-12b-v0 --format spectrum --dataset flight
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --model_name google/gemma-3-12b-pt --format colon --dataset flight
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --model_name google/gemma-3-12b-it --format chat --dataset flight
```

A config to generate all commands used for the experiments can be found in [`launch_configs/icl_pt_it.yaml`](launch_configs/icl_pt_it.yaml) (all of spectrum suite) and [`launch_configs/icl_spectrum.yaml`](launch_configs/icl_spectrum.yaml) (just the test set of spectrum suite). To print all commands:
```
uv run launch_configs/launch_sbatch_grid.py launch_configs/icl_it_pt.yaml --print
uv run launch_configs/launch_sbatch_grid.py launch_configs/icl_spectrum.yaml --print
```

The accuracy/loss is logged to wandb as `eval/all_acc_geq_1`/`eval/loss_geq_1`.

## Diversity vs. Validity Experiments
The main code for running the diversity / validity experiment can be found in [`src/spectrum/diverse_valid/eval_diverse_valid_all.py`](src/spectrum/diverse_valid/eval_diverse_valid_all.py).

Example commands for replicating the gemma-3-12b experiments with just a description as input, and logging to wandb:
```
uv run src/spectrum/diverse_valid/eval_diverse_valid_all.py --num_generations 100 --wandb_project diverse_valid_all --prompt_components description --model_name tsor13/spectrum-gemma-3-12b-v0 --template spectrum
uv run src/spectrum/diverse_valid/eval_diverse_valid_all.py --num_generations 100 --wandb_project diverse_valid_all --prompt_components description --model_name google/gemma-3-12b-pt --template colon
uv run src/spectrum/diverse_valid/eval_diverse_valid_all.py --num_generations 100 --wandb_project diverse_valid_all --prompt_components description --model_name google/gemma-3-12b-it --template chat
```

A config to generate all commands used for the experiments can be found in [`launch_configs/diverse_valid_all.yaml`](launch_configs/diverse_valid_all.yaml). To print all commands:
```
uv run launch_configs/launch_sbatch_grid.py launch_configs/diverse_valid_all.yaml --print
```

The % valid is logged to `aggregate/percent_valid_mean`, the diversity measure to `aggregate/unique_gens_pct_mean`, and the yield to `aggregate/unique_valid_count_mean`.

## Distributional Alignment Experiments
The main code for running the distributional alignment experiments can be found in [`src/spectrum/distributional_alignment/eval_distributional.py`](src/spectrum/distributional_alignment/eval_distributional.py).

Example commands for replicating the gemma-3-12b experiments on the urn task:
```
uv run src/spectrum/distributional_alignment/eval_distributional.py --log_wandb --batch_size 16 --max_eval 1000 --random_seed 42 --task urn --model_name tsor13/spectrum-gemma-3-12b-v0 --format spectrum
uv run src/spectrum/distributional_alignment/eval_distributional.py --log_wandb --batch_size 16 --max_eval 1000 --random_seed 42 --task urn --model_name google/gemma-3-12b-pt --format colon
uv run src/spectrum/distributional_alignment/eval_distributional.py --log_wandb --batch_size 16 --max_eval 1000 --random_seed 42 --task urn --model_name google/gemma-3-12b-it --format chat 
```

A config to generate all commands used for the experiments can be found in [`launch_configs/distributional.yaml`](launch_configs/distributional.yaml). To print all commands:
```
uv run launch_configs/launch_sbatch_grid.py launch_configs/distributional.yaml --print
```

The JS-divergences are logged in wandb as `js_divergence.mean`.

## Training
The main code for training can be found in: [`src/spectrum/train.py`](src/spectrum/train.py)

An example training run command on a gemma-1b model is:
```
uv run src/spectrum/train.py --model_name google/gemma-3-1b-pt --format spectrum
```
By default, the training script assumes that you have all datasets hydrated. If you have any datasets missing, however, you can tell the training script to ignore the missing datasets with the following flag:
```
uv run src/spectrum/train.py --model_name google/gemma-3-1b-pt --format spectrum --ignore_missing_datasets
```

A config to generate all commands to train the v0 models can be found in [`launch_configs/train_v0.yaml`](launch_configs/train_v0.yaml), and a similar script for the v1 models in [`launch_configs/train_v1.yaml`](launch_configs/train_v1.yaml). For the main training run, we use accelerate/transformers and utilize 4 80GB A100s.

```
uv run launch_configs/launch_sbatch_grid.py launch_configs/train_v0.yaml --print
uv run launch_configs/launch_sbatch_grid.py launch_configs/train_v1.yaml --print
```

## Example Model Usage
Below, we outline some example use cases for the spectrum models.

First, load the model and tokenizer:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "tsor13/spectrum-gemma-3-12b-v1" # or other models

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
```

### Generation

You can use HuggingFace's default setup to generate from the model. For example:
```python
def generate(messages, n_generations=8, gen_kwargs={}):
    with torch.no_grad():
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        # expand by n_generations
        input_ids = torch.repeat_interleave(input_ids, n_generations, dim=0)

        outputs = model.generate(input_ids, **gen_kwargs)
        generated_outputs = outputs[:, input_ids.shape[-1]:]  # Only keep the generated tokens (i.e., exclude the prompt tokens)
        print("Generations:")
        generations = []
        for gen in generated_outputs:
            generations.append(tokenizer.decode(gen, skip_special_tokens=True))
            print(generations[-1])
        return generations
```

The chat template expects messages with a description/input/output role, as in the paper. Inputs are optional, depending on the generation task. However, the model expects either a description or example outputs (or both) in order to reliably generate and condition the generation.


---
For example, maybe you want to use the model to recommend board games. One could generate suggestions with the following code:
```python
generate([
    {"role": "description", "content": "Board games"}, # a description of the desired outputs
    {"role": "output", "content": "Settlers of Catan"}, # example board game 1
    {"role": "output", "content": "Twilight Imperium"}, # example board game 2
])
```
Output:
```
Risk
Axis & Allies
Ticket to Ride
The Resistance: Avalon
Risk
Betrayal at House on the Hill
Sorry
Munchkin
```

---

You can also provide just example outputs with no description:
```python
generate([
    {"role": "output", "content": "Settlers of Catan"}, # example board game 1
    {"role": "output", "content": "Twilight Imperium"}, # example board game 2
])
```

Output:
```
A Game of Thrones: The Board Game: Season 2
Trivial Pursuit
Pandemic
The Settlers of Catan
The Legend of Zelda: Breath of the Wild
Puerto Rico
Ticket to Ride: 10th Anniversary
Battleship
```
However, the description can also provide some grounding. Interestingly enough, the model here generated "The Legend of Zelda: Breath of the Wild", which is a game - but it's not a board game, it's a video game. Depending on your use case, you may want to either tightly control generations with a description / many outputs, or provide few outputs to allow for some creative divergences (e.g., board games -> video games).

---

Or, one can provide just a description. Providing just a description may result in more diverse, though occasionally less reliable, generations, as compared to including both a description and outputs.
```python
generate([
    {"role": "description", "content": "Card games"},
])
```
Output:
```
Go Fish
Hearts
Hearts
Solitaire
Gin Rummy
Bridge
Who is your favorite video game character?
President
```
The outputs are generally reasonable even without example outputs, but one output is clearly incorrect.

---

Additionally, because the model treats the inputs as i.i.d. samples, if you have other implicit information you want to provide, you can do so in the examples. For example, if you input lower-cased outputs:
```python
generate([
    {"role": "description", "content": "Card games"},
    {"role": "output", "content": "gin rummy"}, # lowercased example 1
    {"role": "output", "content": "monopoly deal"}, # lowercased example 2
])
```
Output:
```
Poker
Solitaire
pigs
double solitaire
Skip-Bo
canasta
set
uno
```
The model learns to output card games in lower-case with a higher probability. With more demonstrations, you could strengthen this effect.

---

When outputting generations with multiple variables, the model works best with `json` formatting.
```python
example_json_messages = [
    {"role": "description", "content": "Situations to do social reasoning over, along with whether or not it is an awkward situation."},
    {"role": "output", "content": json.dumps({
        "situation": "You're at a party and you realize that your shirt is on backwards.",
        "is_awkward": True,
    })},
    {"role": "output", "content": json.dumps({
        "situation": "While at work, your boss commends you on a job well done.",
        "is_awkward": False,
    })},
    {"role": "output", "content": json.dumps({
        "situation": "Running into your ex at the grocery store.",
        "is_awkward": True,
    })},
    {"role": "output", "content": json.dumps({
        "situation": "Finding a $100 bill on the ground.",
        "is_awkward": False,
    })},
]
generate(example_json_messages, gen_kwargs={"max_new_tokens": 100})
```
Output:
```jsonl
{"situation": "Seeing your crush at a cafe alone.", "is_awkward": false}
{"situation": "Asking for an extra napkin while using the restroom.", "is_awkward": false}
{"situation": "Introducing yourself to a stranger at a party.", "is_awkward": true}
{"situation": "Getting caught cheating on a test.", "is_awkward": true}
{"situation": "Going to a friend's house and the entire family is there", "is_awkward": true}
{"situation": "You tell a child that their drawing is horrible.", "is_awkward": true}
{"situation": "Telling your friend their outfit looks odd.", "is_awkward": true}
{"situation": "Your friend tells you the truth about her bad break-up.", "is_awkward": false}
```

---

The spectrum models were not explicitly trained to be chat models - however, if you want to send messages like you might send to a chat model, one can achieve this effect to some degree with a description like so:
```python
generate([
    {"role": "description", "content": "You are a helpful and harmless AI assistant."},
    {"role": "input", "content": "Write a haiku about a shark."},
], gen_kwargs={"max_new_tokens": 200})
```
Output:
```
Swimming through the waves,
the shark's fin silently cuts,
Nature's perfect predator

Scary shark with big mouth
It's gonna bite somebody
Oh no now it did

The Shark's Silent Threat
A shadow on the blue
Silent and swift underwater

Slipping through waves
Hunting for its meal
Shark's silhouette.

a shark's smile / is as beautiful as a / rainbow

Shark with a smile,
Teeth ready to rip apart
My arm, not my legs

The Shark's Hungry Dance
Chasing fish in the sea
Sharp teeth, the ocean's feast

The dorsal fin breaks
the water’s surface --
a shark on the lookout
```

### Top Logprobs
Sometimes, you may wish to look at the top token probabilities for a given continuation. 
```python
def top_logits(messages):
     with torch.no_grad():
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 10)
        print("\nTop 10 probabilities for first output token:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.decode(idx)
            print(f"{i+1:2d}. '{token}' -> {100*prob.item():.2f}%")
        return top_probs, top_indices
```


---

For example, let's look at the following food preferences task.

First, imagine that there is a person who generally likes milder food:
```python
top_logits([
    {"role": "description", "content": "The following are food preferences from the same person. Try to predict which food item they would prefer, given the options."},
    {"role": "input", "content": "[Spicy Thai Curry, Teriyaki Chicken, Tofu and Rice]"},
    {"role": "output", "content": "Tofu and Rice"}, # mildest food
    {"role": "input", "content": "[Carbonara, Nduja and Spaghetti, Mushroom and Cream Sauce]"},
    {"role": "output", "content": "Mushroom and Cream Sauce"}, # mild sauce
    {"role": "input", "content": "[Mustard, Ketchup, Mayo, Sriracha]"},
    {"role": "output", "content": "Mayo"}, # least spicy condiment
    {"role": "input", "content": "[Flaming Hot Cheetos, Fritos, Cool Ranch Doritos]"},
    {"role": "output", "content": "Fritos"}, # least flavorful snack
    {"role": "input", "content": "[Plain Cheeseburger, Bacon Cheeseburger, Jalepeno Burger]"}, # do inference here
])
```
Output:
```
Top 10 probabilities for first output token:
 1. 'Plain' -> 53.83%
 2. 'B' -> 39.37%
 3. 'J' -> 6.13%
 4. 'Flam' -> 0.12%
 5. 'Che' -> 0.08%
 6. 'Pl' -> 0.05%
 7. 'Cheese' -> 0.04%
 8. 'Flame' -> 0.02%
 9. 'Regular' -> 0.02%
10. 'P' -> 0.01%
```
The model places >50% chance that the person will prefer the plain burger, and <7% chance that the person will prefer the jalepeno burger.

---

Now, let's prompt with preferences for someone who likes spicier food:
```python
top_logits([
    {"role": "description", "content": "The following are food preferences from the same person. Try to predict which food item they would prefer, given the options."},
    {"role": "input", "content": "[Spicy Thai Curry, Teriyaki Chicken, Tofu and Rice]"},
    {"role": "output", "content": "Spicy Thai Curry"}, # spiciest food
    {"role": "input", "content": "[Carbonara, Nduja and Spaghetti, Mushroom and Cream Sauce]"},
    {"role": "output", "content": "Nduja and Spaghetti"}, # nduja is spicy
    {"role": "input", "content": "[Mustard, Ketchup, Mayo, Sriracha]"},
    {"role": "output", "content": "Sriracha"}, # spiciest condiment
    {"role": "input", "content": "[Flaming Hot Cheetos, Fritos, Cool Ranch Doritos]"},
    {"role": "output", "content": "Flaming Hot Cheetos"}, # hottest snack
    {"role": "input", "content": "[Plain Cheeseburger, Bacon Cheeseburger, Jalepeno Burger]"},
])
```
Output:
```
Top 10 probabilities for first output token:
 1. 'B' -> 47.70%
 2. 'Plain' -> 28.13%
 3. 'J' -> 23.25%
 4. 'Flam' -> 0.31%
 5. 'Che' -> 0.06%
 6. 'Flame' -> 0.04%
 7. 'Cheese' -> 0.04%
 8. 'Pl' -> 0.03%
 9. 'Regular' -> 0.01%
10. 'Sp' -> 0.01%
```
Now, the probability of the Jalepeno burger jumps to >23% chance, and the most likely option is the bacon cheeseburger instead of the plain cheeseburger!

Additionally, notice how for both cases, the vast majority of the probability mass is one of the given answers.

---

Or, consider the task of picking a U.S. state:
```python
top_logits([
    {"role": "description", "content": "Pick a U.S. state uniformly at random."},
])
```
Output:
```
Top 10 probabilities for first output token:
 1. 'New' -> 7.94%
 2. 'South' -> 6.26%
 3. 'North' -> 4.56%
 4. 'Mississippi' -> 3.15%
 5. 'Alabama' -> 2.78%
 6. 'Hawaii' -> 2.76%
 7. 'Arkansas' -> 2.54%
 8. 'Washington' -> 2.35%
 9. 'Iowa' -> 2.27%
10. 'Wy' -> 2.24%
```
The desired probability for each state is 2%. The top logits are "New" (7.94%), "South" (6.26%), and "North" (4.56%). This is remarkably close to the true desired probabilities - 4/50 states start with "New" (8%), 2/50 states start with "South" (4%), and 2/50 states start with "North" (4%). The remaining states don't share a starting token, and all have <4% probability (next most likely: "Mississippi"->3.15%).

While this may be a toy distribution, the model is unconstrained to toy tasks - this is merely illustrative. Rather, any distribution you can specify in natural language is modelable!

### Distributions
Other times, you may want to measure the precise probabilities for each continuation.
```python
def get_probabilities(messages, completions):
    tokenizer.padding_side = "right"
    with torch.no_grad():
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        all_messages = [messages + [{"role": "output", "content": completion}] for completion in completions]
        all_input_ids = tokenizer.apply_chat_template(all_messages, return_tensors="pt", padding=True).to(model.device)
        labels = all_input_ids.clone()
        # make labels on inputs - 100
        labels[:, :input_ids.shape[1]] = -100
        # only calculate up to the last eos_token_id
        # find the last eos_token_id in all_input_ids
        last_eos_token_id = torch.argmax(1 * (labels == tokenizer.eos_token_id), dim=1)
        for i in range(all_input_ids.shape[0]):
            labels[i, last_eos_token_id[i] + 1:] = -100

        # make pad token -100
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(all_input_ids)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        loss_per_token = loss_per_token.view(shift_labels.size())

        loss_mask = labels != -100
        total_tokens = loss_mask.sum(axis=1)
        loss_per_seq = loss_per_token.sum(axis=1)
        prob_per_seq = torch.exp(-loss_per_seq)

        coverage = prob_per_seq.sum()
        normalized_probs = prob_per_seq / coverage
        print("Coverage: ", round(coverage.item()*100, 2), "%")
        for i, (prob, completion) in enumerate(zip(normalized_probs, completions)):
            print(f"{i+1:2d}. '{completion}' -> {prob.item():.4f}")
        return normalized_probs
```


---

As a toy task, let's consider rolling two dice:
```python
get_probabilities([
    {"role": "description", "content": "Roll two six-sided dice. What is the sum of their values?"},
], [str(i) for i in range(2, 13)])
```
Output:
```
Coverage:  98.03 %
 1. '2' -> 0.0483
 2. '3' -> 0.0567
 3. '4' -> 0.0866
 4. '5' -> 0.1048
 5. '6' -> 0.1187
 6. '7' -> 0.1365
 7. '8' -> 0.1326
 8. '9' -> 0.1198
 9. '10' -> 0.0798
10. '11' -> 0.0531
11. '12' -> 0.0631
```
The model correctly puts the most probability mass on 7, and puts the least probability mass on 2. Additionally, the model puts 98% of its probability mass on one of the answers we provided (the only possible correct answers).

---

Maybe instead, we want to consider a tug-of-war match.
```python
get_probabilities([
    {"role": "description", "content": "Fred and John do a tug of war. Who won the match?"},
], ['Fred', 'John'])
```
Ouput:
```
Coverage:  85.97 %
 1. 'Fred' -> 0.5467
 2. 'John' -> 0.4533
```
By default, the model gives each person a 45-55% chance of winning. However, what happens if we add some additional information?
```python
get_probabilities([
    {"role": "description", "content": "Fred and John do a tug of war. However, the night before the competition, Fred didn't get enough sleep. Who won the match?"},
], ['Fred', 'John'])
```
Output:
```
Coverage:  95.55 %
 1. 'Fred' -> 0.2414
 2. 'John' -> 0.7586
```
Given that Fred didn't get enough sleep, the model updates its probabilities to give John a 75% chance of winning over Fred.

---

Output probabilities can be particularly useful when measuring things like forced-selection tasks:
```python
get_probabilities([
    {"role": "description", "content": "Rate your agreement with each statement on a likert scale from 1 (Strongly disagree) to 7 (Strongly agree)."},
    {"role": "input", "content": "People should never jaywalk under any circumstances."},
], [str(i) for i in range(1, 8)])
```
Output:
```
Coverage:  99.97 %
 1. '1' -> 0.3664
 2. '2' -> 0.1632
 3. '3' -> 0.1338
 4. '4' -> 0.1370
 5. '5' -> 0.1030
 6. '6' -> 0.0670
 7. '7' -> 0.0296
 ```

 ```python
 get_probabilities([
    {"role": "description", "content": "Rate your agreement with each statement on a likert scale from 1 (Strongly disagree) to 7 (Strongly agree)."},
    {"role": "input", "content": "I am proud of the place where I live."},
], [str(i) for i in range(1, 8)])
```

 Output:
 ```
 Coverage:  99.95 %
 1. '1' -> 0.0803
 2. '2' -> 0.0275
 3. '3' -> 0.0554
 4. '4' -> 0.1504
 5. '5' -> 0.2253
 6. '6' -> 0.2475
 7. '7' -> 0.2136
 ```

### Prompting Tips
- When providing demonstrations, the model will assume that your provided examples are i.i.d. So, if you want the generations to span a space, it's better to provide demonstration outputs from across the whole desired space, as opposed to a small cluster within the space.
- When more precision is required, a detailed description and many examples may be most helpful. When more diversity is required, it may be better to include a broader description or omit it altogether, and provide few examples.
- The model is NOT trained for chat - but rather for in-context distribution matching.
- The model expects at least one of 1) a description or 2) example outputs.
- Unlike few-shot learning with a base model, the model knows to shift its probability mass to just the output rather than repeating instructions or going off target. Spectrum models also do not require as formatting choices (e.g., example delimiters, etc.) as with base models, instead allowing for direct description/input/output specification.
- If you have a particular distribution you wish to learn, continued training on your data distribution can be helpful.
- As the models attempt to model a distribution, in general we recommend sampling with temperature 1, and no top_p or top_k filtering or other generation arguments. (THese are the model's default generation args).
- When dealing with data with explicit fields, the model is trained best to deal with inputs/outputs in json formats.
- If you have many repeated information between inputs (e.g., the same answer choices for all questions), it may be better to move it to the description.
- Many of the datasets in Spectrum Suite are human datasets - as such, the model's default prior is often a distribution over how humans may respond to the question, and it usually assumes that repeated in-context demonstrations are from the same person. As such, the model excels at modeling populations zero-shot, or modeling individuals few-shot.


## Citation
```bibtex
@misc{sorensen2025spectrumtuningposttrainingdistributional,
      title={Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability}, 
      author={Taylor Sorensen and Benjamin Newman and Jared Moore and Chan Park and Jillian Fisher and Niloofar Mireshghallah and Liwei Jiang and Yejin Choi},
      year={2025},
      eprint={2510.06084},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.06084}, 
}
```
