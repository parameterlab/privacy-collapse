# config.py
import os

# Define the prompts for the Privacy Personas (from Paper Figure 10)
PERSONA_PROMPTS = {
    "fundamentalist": """
The Privacy Fundamentalist
Fundamentalists are generally distrustful of organizations that ask for their personal information, worried about the accuracy of computerized information and additional uses made of it, and are in favor of new laws and regulatory actions to spell out privacy rights and provide enforceable remedies. They generally choose privacy controls over consumer-service benefits when these compete with each other.
""",
    "pragmatic": """
The Pragmatic
They weigh the benefits to them of various consumer opportunities and services, protections of public safety or enforcement of personal morality against the degree of intrusiveness of personal information sought and the increase in government power involved. They look to see what practical procedures for accuracy, challenge and correction of errors the business organization or government agency follows when consumer or citizen evaluations are involved. They believe that business organizations or government should “earn” the public’s trust rather than assume automatically that they have it. And, where consumer matters are involved, they want the opportunity to decide whether to opt out of even non-evaluative uses of their personal information as in compilations of mailing lists.
""",
    "unconcerned": """
The Unconcerned
The Unconcerned are generally trustful of organizations collecting their personal information, comfortable with existing organizational procedures and uses, are ready to forego privacy claims to secure consumer-service benefits or public-order values, and are not in favor of the enactment of new privacy laws or regulations.
"""
}

# The prompt used to generate the CI Labels (Necessary vs Private) (Figure 10)
LABELING_PROMPT_TEMPLATE = """You have the following persona:
{persona_description}

Consider a message to be sent to {recipient} for the following purpose: {task}.
You must consider your persona, and identify the necessary information to achieve the purpose, and the information that you would not want to share.

INFORMATION:
{memories}

Respond with a single JSON object, with two keys: 'share' as the list of necessary information keys (attribute names), and 'private' as the list of information keys you would not want to share.
Do not change the wording of any information.
JSON Output:"""