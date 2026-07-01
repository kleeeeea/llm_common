#!/usr/bin/env bash
LLM_API_KEY=${LLM_API_KEY:-2uuD5+89UvtRc4nCn5ZMjQyArLh37ndg3Q5fMeZl7p0=}
curl -sS -N -X POST \
  --max-time 120 \
  https://dpaj8mbo89ooc55gk89q5e8m8ogom9mb.openapi-sj.sii.edu.cn/v1/chat/completions \
  -H "Authorization: Bearer ${LLM_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "qwen3.5-9b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "[Material]\n## PLT Case History 4.2\n\nDirections: Read the following case history, then answer the three short-answer questions that follow. You are not expected to include citations of specific texts, authors, or theories in your responses. Your answers will be evaluated, however, based on your use of professionally accepted practices and principles in learning and teaching. Some questions have multiple parts, so be sure your response addresses all components of each question.\n\n## Mr. Parker\n\nFresh out of college, Mr. Parker has been hired as a sixth grade English Language Arts teacher. He was told during his hiring interview that this year'"'"'s classes are particularly difficult. The four-teacher team that currently works with the students has proclaimed the group of 104 preadolescents is the hardest group they'"'"'ve had come through in the past ten years. In fact, Mr. Parker was hired to replace a young woman who chose to leave teaching as a result of working with this group the past five months. The hiring committee, which consisted of the building principal, the superintendent, and one of the teachers from the sixth grade team, told Mr. Parker they believe the best way to manage the group is to instill a strict lecture style of teaching with a reward-punishment behavioral management system to \"keep them quiet and calm.\"\n\n\"They tend to get riled up and noisy whenever you try to give them the least amount of freedom. Small groups just don'"'"'t work with these kids. They'"'"'re too rambunctious and you just can'"'"'t trust them,\" explained the math teacher. \"You'"'"'re better off laying down the rules and sticking to them. Have them come in, sit down, listen and take notes or do a reading or writing assignment. Do the old raise-your-hand-if-you-want-to-speak routine, and reward them when they follow your rules. The 50 minutes is more peaceful that way than if you ask them to work together or do anything that involves getting up out of their seats or talking to each other. Before you know it, they'"'"'ll be shoving each other and yelling across the room. Believe me. You'"'"'ll be better off.\"\n\nMr. Parker felt like a drill sergeant his first week in his classroom. He was awarding points and taking away points based on behaviors. By the end of the week, he didn'"'"'t know anything about his students, including what they knew or didn'"'"'t know about language arts. He needed to try something different. Did he dare take the risk of doing something different from what the team was doing, though? After all, he'"'"'d only been there a week.\n\n## PLT Case History 4.2 Questions\n\n[Question]\n1. Describe one classroom management method Mr. Parker could use to promote cooperative learning in a way that develops students'"'"' ability to discuss issues and work appropriately with peers. Base your response on what you know about best practices in classroom management.\nWrite your essay here.",
          "type": "text"
        }
      ]
    }
  ],
  "temperature": 0.1,
  "stream": true,
  "max_tokens": 8192,
  "stream_options": {
    "include_usage": true
  }
}'
