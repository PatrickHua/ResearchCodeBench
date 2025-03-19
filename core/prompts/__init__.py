# from core.prompts.rubric_dynamic import RUBRIC_GENERATION_PROMPT
# from core.prompts.rubric import GRADING_PROMPT




if __name__ == "__main__":
    code = "def random_masking(self, x, mask_ratio):\n    \"\"\"\n        Perform per-sample random masking by per-sample shuffling.\n        Per-sample shuffling is done by argsort random noise.\n        x: [N, L, D], sequence\n        \"\"\"\n    N, L, D = x.shape\n    len_keep = int(L * (1 - mask_ratio))\n    noise = torch.rand(N, L, device=x.device)\n    ids_shuffle = torch.argsort(noise, dim=1)\n    ids_restore = torch.argsort(ids_shuffle, dim=1)\n    ids_keep = ids_shuffle[:, :len_keep]\n    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))\n    mask = torch.ones([N, L], device=x.device)\n    mask[:, :len_keep] = 0\n    mask = torch.gather(mask, dim=1, index=ids_restore)\n    return (x_masked, mask, ids_restore)"
    completions = "import torch.nn.functional as F\n\n# Complete the function\ndef random_masking(self, x, mask_ratio):\n    \"\"\"\n    Perform per-sample random masking by per-sample shuffling.\n    Per-sample shuffling is done by argsort random noise.\n    x: [N, L, D], sequence\n    \"\"\"\n    assert x.size(0) == self.num_patches + 1  # for cls_token\n    assert mask_ratio <= 1 and mask_ratio >= 0  # constant val\n\n    # Hardy's trick\n    mask_shape = [x.size(0), 1] + [1 for _ in range(len(x.size()) - 1)]\n    mask_shape = tuple(mask_shape)\n\n    mask = torch.bernoulli(mask_ratio * torch.ones(mask_shape, device=x.device, dtype=x.dtype))\n    x_masked = x.clone()\n\n    n = mask.sum(dim=1).to(torch.long)\n\n    for i in range(self.decoder_depth):\n        for j in range(n[i].shape[0] // self.decoder_depth):  # process masking by step\n            mask[i, n[i][j * self.decoder_depth:(j + 1) * self.decoder_depth]] = 1\n\n    return mask"

    print(RUBRIC_GENERATION_PROMPT.format(code=code))


    import openai
    import os
    import json
    from core.utils import parse_grade
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": "You are an expert educator."},
    #         {"role": "user", "content": RUBRIC_GENERATION_PROMPT.format(code=code)}
    #     ]
    # )

    # # print(response.choices[0].message.content)
    # rubric = response.choices[0].message.content
    # print(rubric)

    # # save it to a file
    # with open('rubric.txt', 'w') as f:
    #     f.write(rubric)
    # # read it back
    # with open('rubric.txt', 'r') as f:
    #     rubric = f.read()

    # breakpoint()

    # total_points = parse_grade(rubric, patterns=['total_points'])[0]
    # print(f"Total Points: {total_points}")


    # breakpoint()
    
    prompt = GRADING_PROMPT.format(reference_code=code, student_code=completions)
    # print(prompt)
    # breakpoint()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert educator."},
            {"role": "user", "content": prompt}
        ]
    )

    # print(response.choices[0].message.content)
    grade = response.choices[0].message.content

    # save it to a file
    with open('grade.txt', 'w') as f:
        f.write(grade)

    # read it back
    with open('grade.txt', 'r') as f:
        grade = f.read()

    
    functionality, logic, semantic_similarity, code_quality = parse_grade(grade, patterns=['functionality', 'logic', 'semantic_similarity', 'code_quality'])
    print(f"Functionality: {functionality}, Logic: {logic}, Semantic Similarity: {semantic_similarity}, Code Quality: {code_quality}")
