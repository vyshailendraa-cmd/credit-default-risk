# Packages the API code and the deployment model into function.zip for AWS Lambda.
# Run from the project root:   powershell -ExecutionPolicy Bypass -File build_lambda_zip.ps1
#
# Uses tar (built into Windows 10+) rather than Compress-Archive, because Compress-Archive
# writes folder paths with backslashes, which Lambda (Linux) does not treat as folders.
Remove-Item -Recurse -Force lambda_pkg, function.zip -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path lambda_pkg\models | Out-Null
Copy-Item api\app.py, api\index.html, features.py lambda_pkg\
Copy-Item models\slim_model.json, models\slim_schema.json lambda_pkg\models\
# Lambda's default handler is lambda_function.lambda_handler; point it at the API.
Set-Content lambda_pkg\lambda_function.py "from app import handler as lambda_handler"
tar -a -c -f function.zip -C lambda_pkg lambda_function.py app.py features.py index.html models
Remove-Item -Recurse -Force lambda_pkg
Get-Item function.zip | Select-Object Name, Length
