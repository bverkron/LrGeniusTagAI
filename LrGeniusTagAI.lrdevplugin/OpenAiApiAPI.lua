OpenAiApiAPI = {}
OpenAiApiAPI.__index = OpenAiApiAPI

-- Lists the models exposed by a user-run, OpenAI API compatible server
-- (e.g. llama.cpp server, mlx_vlm.server, mlx_lm.server, vLLM,
-- text-generation-webui, LocalAI, ...) at prefs.openaiApiUrl. Unlike
-- Ollama/LM Studio, the standard /v1/models response does not reliably
-- report vision capability across server implementations, so every
-- returned model is listed here; make sure the model you pick actually
-- supports image input.
function OpenAiApiAPI.getLocalVisionModels()
    local response, headers = LrHttp.get(prefs.openaiApiUrl .. Defaults.openaiApiListModelUrl)

    if headers.status == 200 then
        if response ~= nil then
            log:trace(response)
            local decoded = JSON:decode(response)
            if decoded ~= nil then
                local models = {}
                if decoded.data ~= nil and type(decoded.data) == "table" then
                    for _, model in ipairs(decoded.data) do
                        local name = model.id
                        log:trace("Found OpenAI API model: " .. name)
                        table.insert(models, { title = "OpenAI API " .. name , value = 'openaiapi-' .. name })
                    end
                end
                return models
            end
        else
            log:error('Got empty response from OpenAiApiAPI')
        end
    else
        log:error('OpenAiApiAPI GET request failed. ' .. prefs.openaiApiUrl .. Defaults.openaiApiListModelUrl)
        log:error(Util.dumpTable(headers))
        log:error(response)
        return nil
    end
    return nil
end

function OpenAiApiAPI:new()
    local o = setmetatable({}, OpenAiApiAPI)

    self.model = string.sub(prefs.ai, 11, -1)
    self.url = prefs.openaiApiUrl
    self.chatUrl = self.url .. Defaults.openaiApiChatUrl

    return o
end

function OpenAiApiAPI:doRequest(filePath, task, systemInstruction, generationConfig)
    local body = {
        model = self.model,
        response_format = generationConfig,
        messages = {
            {
                role = "system",
                content = systemInstruction,
            },
            {
                role = "user",
                content = {
                    { type = "text", text = task },
                    {
                        type = "image_url",
                        image_url = {
                            url = "data:image/jpeg;base64," .. Util.encodePhotoToBase64(filePath)
                        }
                    }
                }
            }
        },
        temperature = prefs.temperature,
    }

    log:trace(Util.dumpTable(body))

    local response, headers = LrHttp.post(self.chatUrl, JSON:encode(body), {{ field = 'Content-Type', value = 'application/json' }}, 'POST', 720)

    if headers.status == 200 then
        if response ~= nil then
            log:trace(response)
            local decoded = JSON:decode(response)
            if decoded ~= nil then
                if decoded.choices ~= nil then
                    local inputTokenCount = 0
                    local outputTokenCount = 0
                    if decoded.usage ~= nil then
                        inputTokenCount = decoded.usage.prompt_tokens or 0
                        outputTokenCount = decoded.usage.completion_tokens or 0
                    end
                    if decoded.choices[1].finish_reason == 'stop' then
                        local text = decoded.choices[1].message.content
                        log:trace(text)
                        return true, text, inputTokenCount, outputTokenCount
                    else
                        ErrorHandler.handleError('OpenAI API request failed', 'Finish reason: ' .. decoded.choices[1].finish_reason)
                        return false, decoded.choices[1].finish_reason, inputTokenCount, outputTokenCount
                    end
                else
                    ErrorHandler.handleError('OpenAI API request failed', Util.dumpTable(decoded))
                    return false, 'OpenAiApiAPI POST request failed. No choices in response', 0, 0
                end
            else
                ErrorHandler.handleError('OpenAI API request failed', 'Response from OpenAI API could not be decoded: ' .. response)
                return false, 'Response from OpenAI API could not be decoded', 0, 0
            end
        else
            ErrorHandler.handleError('OpenAI API request failed', 'Got empty response from OpenAI API')
            return false, 'OpenAiApiAPI request failed: Got empty response', 0, 0
        end
    else
        ErrorHandler.handleError('OpenAiApiAPI POST request failed', 'HTTP headers: ' .. Util.dumpTable(headers) .. ' HTTP Response: ' .. (response or 'nil'))
        return false, 'OpenAiApiAPI POST request failed. ' .. self.chatUrl, 0, 0
    end
end


function OpenAiApiAPI:analyzeImage(filePath, metadata)
    local task = AiModelAPI.generatePromptFromConfiguration()
    if metadata ~= nil then
        if prefs.submitGPS and metadata.gps ~= nil then
            task = task .. " " .. "\nThis photo was taken at the following coordinates:" .. metadata.gps.latitude .. ", " .. metadata.gps.longitude
        end
        if prefs.submitKeywords and metadata.keywords ~= nil then
            task = task .. " " .. "\nSome keywords are:" .. metadata.keywords
        end
        if metadata.context ~= nil and metadata.context ~= "" then
            log:trace("Preflight context given")
            task = task .. "\nSome context for this photo: " .. metadata.context
        end
        if metadata.folderNames ~= nil and prefs.submitFolderName and string.find(metadata.folderNames, "%a") then
            log:trace("Submit folder names enabled")
            task = task .. "\nThis photo is located in the following folders: " .. metadata.folderNames
        end
    end

    local systemInstruction = AiModelAPI.addKeywordHierarchyToSystemInstruction()

    local success, result, inputTokenCount, outputTokenCount = self:doRequest(filePath, task, systemInstruction, ResponseStructure:new():generateResponseStructure())
    if success then
        if prefs.replaceSS then
            result = string.gsub(result, "ß", "ss")
        end
        return success, JSON:decode(result), inputTokenCount, outputTokenCount
    end
    return false, "", inputTokenCount, outputTokenCount
end
