curl \
  --request POST \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer ${IAM_TOKEN}" \
  --data "@prompt_custom.json" \
  "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"