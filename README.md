# Cookbook: MCP-Google-Map-Agent
 **In this step-by-step tutorial, you’ll build a fully serverless AI agent that taps into Google Maps through the MCP platform. You’ll learn how to:**
> 1. **Connect** securely to the Google Maps MCP endpoint.  
> 2. **Retrieve** location data on-demand without managing any servers.  
> 3. **Summarize** the results with a LLM to deliver concise, human-readable insights.

## Architecture
![MCP-Google-Maps](https://github.com/user-attachments/assets/0945de35-53b1-4caf-ba1d-c3667aba7cd4)

---
## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after forking.
- **Python>=3.8**. You would need Python to customize the code in the app.py according to your needs.
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.

---
## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **inferless-runtime-config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the `Deploy a Model` button.

- Select `Github` as the method of upload from the Provider list and then select your Github Repository and the branch.
- Choose the type of machine, and specify the minimum and maximum number of replicas for deploying your model.
- Configure Custom Runtime ( If you have pip or apt packages), choose Volume, Secrets and set Environment variables like Inference Timeout / Container Concurrency / Scale Down Timeout
- Once you click “Continue,” click Deploy to start the model import process.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
          --header 'Content-Type: application/json' \
          --header 'Authorization: Bearer <your_api_key>' \
          --data '{
                  "inputs": [
                              {
                                  "name": "user_query",
                                  "shape": [
                                      1
                                  ],
                                  "data": [
                                      "Can you find me some Pizza shop in Jorhat Gar ali with good number of reviews?"
                                  ],
                                  "datatype": "BYTES"
                              }
                          ]  
                }
            '
```
---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](#input) for more.

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu.

For more information refer to the [Inferless docs](https://docs.inferless.com/).
