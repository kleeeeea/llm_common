#!/usr/bin/env bash
LLM_API_KEY=${LLM_API_KEY:-sk-HBBSahVJVMWeGRttJxhYpwhLYJLNV0IE0OGc0r6IxuKKSGpc}
curl -sS -N -X POST \
  --max-time 120 \
  https://api.innospark.cn/v1/chat/completions \
  -H "Authorization: Bearer ${LLM_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gemini-3.1-pro-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "[Material]\n## PLT Case History 6.2\n\nDirections: Read the following case history, then answer the three short-answer questions that follow. You are not expected to include citations of specific texts, authors, or theories in your responses. Your answers will be evaluated, however, based on your use of professionally accepted practices and principles in learning and teaching. Some questions have multiple parts, so be sure your response addresses all components of each question.\n\n## Keanna Petersen\n\nKeanna is an exchange student from Germany. She will be spending the school year in the small town of Schoharie, and she will be attending classes at the high school. Keanna has been speaking English since she was a young child, but she is finding speaking casual English and being expected to learn through the English language are two different things. She is finding it difficult to learn complex concepts by reading English texts and participating in class activities and discussions. The content-area subjects contain vocabulary that is new to her. For example, in history class, she has never heard many of the American names and places. She gets confused trying to catch up with her classmates even though they are quite amiable and try to help.\n\nKeanna will be returning to Germany when the school year is over, so she just has to do her best to be part of the classes and to learn what she can. She is motivated to learn but does not feel too stressed out to get high grades. She is enjoying the experience and the cultural aspects of being a high schooler in the United States. Grades are secondary to her.\n\nKeanna'"'"'s teacher Mr. Elkin is concerned about Keanna'"'"'s self-motivation. The other students in the class seem to reflect what Mr. Elkin has identified as Keanna'"'"'s laissez-faire attitude about her grades. She is well behaved in class and is very personable; however, she often comes to class unprepared. She doesn'"'"'t always do the required readings and homework assignments.\n\n## PLT Case History 6.2 Questions\n\n[Question]\n2. Generally, research evidence indicates adolescents'"'"' motivation declines in the middle- and high-school years. Explain why this decline frequently occurs.\nWrite your essay here.",
          "type": "text"
        }
      ]
    }
  ],
  "temperature": 0.1,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}'
