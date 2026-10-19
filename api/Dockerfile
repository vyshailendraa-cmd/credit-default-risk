# AWS Lambda container image for the credit default risk API.
# Build from the PROJECT ROOT (not the api folder):
#   docker build --platform linux/amd64 --provenance=false -f api/Dockerfile -t credit-risk-api .
FROM public.ecr.aws/lambda/python:3.12

COPY api/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY features.py api/app.py ${LAMBDA_TASK_ROOT}/
COPY models/slim_model.json models/slim_schema.json ${LAMBDA_TASK_ROOT}/models/

CMD ["app.handler"]
