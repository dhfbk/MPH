#define a class "prompt"
class infoPrompt:
    def __init__(self, scenario_description, input_elements, task_definition, user_space_description, input_format, instruction, output_format, command, output_template):
        self.scenario_description = scenario_description
        self.input_elements = input_elements
        self.task_definition = task_definition
        self.user_space_description = user_space_description
        self.input_format = input_format
        self.instruction = instruction
        self.output_format = output_format
        self.command = command
        self.output_template = output_template


class PromptLlamaMPC:
    def __init__(self, info):
        self.scenario_description = info.scenario_description
        self.input_elements = info.input_elements
        self.task_definition = info.task_definition
        self.user_space_description = info.user_space_description
        self.input_format = info.input_format
        self.instruction = info.instruction
        self.output_format = info.output_format
        self.command = info.command
        self.output_template = info.output_template

    def create_system_instruction(self, conversation=False, interaction=False, summary=False, description=False, response_selection=False, addressee_recognition=False, summarization=False, describe_next_speaker=False):
        scenario_description = self.scenario_description
        input_elements = self.input_elements['general_statement']
        task_definition = ""
        user_space_description = self.user_space_description
        input_format = ""
        instruction = ""
        output_format = ""

        if conversation:
            input_elements = input_elements + "\n\n" + self.input_elements['conversation']
            input_format = input_format + "\n\n" + self.input_format['conversation']

        if interaction:
            input_elements = input_elements + "\n\n" + self.input_elements['interaction']
            input_format = input_format + "\n\n" + self.input_format['interaction']

        if summary:
            input_elements = input_elements + "\n\n" + self.input_elements['summary']
            input_format = input_format + "\n\n" + self.input_format['summary']


        if description:
            input_elements = input_elements + "\n\n" + self.input_elements['description']
            input_format = input_format + "\n\n" + self.input_format['description']



        input_format = input_format[2:]

        if response_selection:
            task_definition = self.task_definition['response_selection']
            instruction = self.instruction['response_selection']
            output_format = self.output_format['response_selection']
        else:
            if addressee_recognition:
                task_definition = self.task_definition['addressee_recognition']
                instruction = self.instruction['addressee_recognition']
                output_format = self.output_format['addressee_recognition']
            else:
                if summarization:
                    task_definition = self.task_definition['summarization']
                    instruction = self.instruction['summarization']
                    output_format = self.output_format['summarization']
                else:
                    if describe_next_speaker:
                        task_definition = self.task_definition['describe_next_speaker']
                        instruction = self.instruction['describe_next_speaker']
                        output_format = self.output_format['describe_next_speaker']

        system_instruction = ("<<SYS>>\n\n" +
                              scenario_description + "\n\n" +
                              input_elements + "\n\n" +
                              task_definition + "\n\n" +
                              user_space_description + "\n\n" +
                              input_format + "\n\n" +
                              instruction + "\n\n" +
                              output_format + "\n\n<</SYS>>")

        return system_instruction

    def create_input(self, item, conversation=False, interaction=False, summary=False, description=False):

        item_input = ""

        if conversation:
            item_input = item_input + "[CONVERSATION]\n" + item['conversation'] + "\n[/CONVERSATION]\n\n"

        if interaction:
            item_input = item_input + "[INTERACTION]\n" + item['interaction'] + "\n[/INTERACTION]\n\n"

        if summary:
            item_input = item_input + "[SUMMARY]\n" + item['summary'] + "\n[/SUMMARY]\n\n"

        if description:
            item_input = item_input + "[DESCRIPTION]\n" + item['description'] + "\n[/DESCRIPTION]\n\n"

        return item_input

    def response_selection(self, conversation=False, interaction=False, summary=False, description=False, item = ""):

        system_instruction = self.create_system_instruction(conversation=conversation, interaction=interaction, summary=summary, description=description, response_selection=True)

        if item == "":
            return system_instruction

        else:
            item_input = self.create_input(item, conversation=conversation, interaction=interaction, summary=summary, description=description)
            prompt = "<s>[INST]" + system_instruction + "\n\n" + item_input + self.command['response_selection'] + "\n\n" + "[/INST]\n\n" + self.output_template['response_selection']

            return prompt, system_instruction

    def addressee_recognition(self, conversation=False, interaction=False, summary=False, description=False, item = ""):

        system_instruction = self.create_system_instruction(conversation=conversation, interaction=interaction, summary=summary, description=description, addressee_recognition=True)

        if item == "":
            return system_instruction

        else:
            item_input = self.create_input(item, conversation=conversation, interaction=interaction, summary=summary, description=description)
            prompt = "<s>[INST]" + system_instruction + "\n\n" + item_input + self.command['addressee_recognition'] + "\n\n" + "[/INST]\n\n" + self.output_template['addressee_recognition']

            return prompt, system_instruction

    def summarization(self, item = ""):

        system_instruction = self.create_system_instruction(conversation=True, interaction=True, summarization=True)

        if item == "":
            return system_instruction

        else:
            item_input = self.create_input(item, conversation=True, interaction=True)
            prompt = "<s>[INST]" + system_instruction + "\n\n" + item_input + self.command['summarization'] + "\n\n" + "[/INST]"

            return prompt, system_instruction

    def describe_next_speaker(self, item = ""):

        system_instruction = self.create_system_instruction(conversation=True, interaction=True, describe_next_speaker=True)

        if item == "":
            return system_instruction

        else:
            item_input = self.create_input(item, conversation=True, interaction=True)
            prompt = "<s>[INST]" + system_instruction + "\n\n" + item_input + self.command['describe_next_speaker'] + "\n\n" + "[/INST]"

            return prompt, system_instruction