from typing import Type
from user_input_process import user_input_init
import pandas as pd
import os

# FIX PATH (quan trọng)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ================================
# FEW-SHOT PROMPT
# ================================
class Prompt_Fewshot:
    def __init__(self, user_input: Type[user_input_init], foundqadf):
        self.user_input_cls = user_input
        self.foundqadf = foundqadf

    def compile_examples(self):
        examples = ""

        for i in range(self.foundqadf.shape[0]):
            q = 'Question: ' + str(self.foundqadf['question'].iloc[i])
            a = 'SQL: ' + str(self.foundqadf['sql'].iloc[i])

            # FIX BUG: thêm newline rõ ràng
            examples += '\n'.join([q, a]) + '\n\n'

        return examples

    def compile(self):
        examples = self.compile_examples()
        hints = self.user_input_cls.gen_hints()
        user_question = self.user_input_cls.input

        prompt = f"""
#Character#
You are an expert SQL developer.

#Task#
Write a SQL query based on similar examples.

#Rules#
- Must follow SQL grammar
- Use hints if provided
- Output only SQL

#Examples#
{examples}

#Now Write SQL#
Question: {user_question}
HINTS: {hints}
SQL:
        """

        return prompt.strip()


# ================================
# ZERO-SHOT PROMPT
# ================================
class Prompt_Zeroshot:
    def __init__(self, user_input: Type[user_input_init]):
        self.user_input_cls = user_input

        # FIX PATH
        self.meta = pd.read_csv(
            os.path.join(BASE_DIR, 'backend', 'dbmata.csv'),
            index_col=0
        )

    def indentify_table_fields(self):

        fields = set()

        # ===== 1. từ hints =====
        hintslist = self.user_input_cls.hintslist
        if len(hintslist) > 0:
            fields |= set([h[1] for h in hintslist])

        # ===== 2. từ text =====
        words = self.user_input_cls.input.lower().split()

        for col in self.meta['Column']:
            col_str = str(col).lower()

            for w in words:
                if w in col_str:
                    fields.add(col)

        return list(fields)

    def compile_meta(self):

        fields = self.indentify_table_fields()

        # DEBUG nhẹ
        print("Fields found:", fields)

        # filter meta
        submeta = self.meta[self.meta['Column'].isin(fields)]

        tables = set(submeta['Table'])

        if len(tables) == 0:
            return ""

        # lấy full schema của table
        submeta = self.meta[self.meta['Table'].isin(tables)]

        schema = []

        for table in tables:

            sub = self.meta[self.meta['Table'] == table].fillna('')

            tabledes = ' '.join(set(sub['Table Description']))

            schema.append(f"Table: {table}")
            schema.append(f"Comment: {tabledes}")
            schema.append("Fields:")

            for i in range(sub.shape[0]):
                field = sub.iloc[i]['Column']
                desc = sub.iloc[i]['Column Description']

                schema.append(f"- {field}: {desc}")

            schema.append("")

        return '\n'.join(schema)

    def compile(self):

        user_question = self.user_input_cls.input
        hints = self.user_input_cls.gen_hints()

        if len(hints) < 5:
            hints = 'No hints'

        schema = self.compile_meta()

        if len(schema) < 5:
            schema = 'No schema found. Write SQL based on general knowledge.'

        prompt = f"""
#Character#
You are an expert SQL engineer.

#Task#
Write a SQL query using schema.

#Rules#
- Use schema fields
- Use hints if available
- Output only SQL

#Schema#
{schema}

#Now Write SQL#
Question: {user_question}
HINTS: {hints}
SQL:
"""

        return prompt.strip()


# ================================
# ANSWER SUMMARIZER
# ================================
class Prompt_Answer:
    def __init__(self, taskinfos) -> None:
        self.taskinfos = taskinfos

    def complie(self):

        prompt = f"""
Summarize the SQL process in <= 40 words.

Question: {self.taskinfos['question']}
SQL: {self.taskinfos['sql']}
Result: {self.taskinfos['sqlexe']}
"""

        return prompt.strip()


# ================================
# TEST MAIN
# ================================
if __name__ == "__main__":

    # TEST INPUT
    user_input = 'How many Booking records are there?'

    full_processed_input = user_input_init(user_input).full_process()

    # TEST ZERO-SHOT
    p = Prompt_Zeroshot(full_processed_input).compile()

    print("\n=== PROMPT ===\n")
    print(p)