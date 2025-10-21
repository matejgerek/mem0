import OpenAI from "openai";
import { LLM, LLMResponse } from "./base";
import { LLMConfig, Message } from "../types";

export class OpenAILLM implements LLM {
  private openai: OpenAI;
  private model: string;
  private reasoningEffort?: "minimal" | "low" | "medium" | "high";

  constructor(config: LLMConfig) {
    const defaultHeaders =
      (config as any).headers ||
      (config.modelProperties?.headers as Record<string, string> | undefined);
    this.openai = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
      ...(defaultHeaders && { defaultHeaders }),
    });
    this.model = config.model || "gpt-4.1-nano-2025-04-14";
    // prefer top-level config.reasoningEffort, fallback to modelProperties.reasoningEffort
    this.reasoningEffort =
      (config as any).reasoningEffort ||
      (config.modelProperties?.reasoningEffort as
        | "minimal"
        | "low"
        | "medium"
        | "high"
        | undefined);
  }

  private normalizeReasoningEffort(): "low" | "medium" | "high" | undefined {
    if (!this.reasoningEffort) return undefined;
    if (this.reasoningEffort === "minimal") return "low";
    return this.reasoningEffort;
  }

  async generateResponse(
    messages: Message[],
    responseFormat?: { type: string },
    tools?: any[],
  ): Promise<string | LLMResponse> {
    const normalizedReasoningEffort = this.normalizeReasoningEffort();
    const completion = await this.openai.chat.completions.create({
      messages: messages.map((msg) => {
        const role = msg.role as "system" | "user" | "assistant";
        return {
          role,
          content:
            typeof msg.content === "string"
              ? msg.content
              : JSON.stringify(msg.content),
        };
      }),
      model: this.model,
      response_format: responseFormat as { type: "text" | "json_object" },
      ...(normalizedReasoningEffort && {
        reasoning_effort: normalizedReasoningEffort,
      }),
      ...(tools && { tools, tool_choice: "auto" }),
    });

    const response = completion.choices[0].message;

    if (response.tool_calls) {
      return {
        content: response.content || "",
        role: response.role,
        toolCalls: response.tool_calls.map((call) => ({
          name: call.function.name,
          arguments: call.function.arguments,
        })),
      };
    }

    return response.content || "";
  }

  async generateChat(messages: Message[]): Promise<LLMResponse> {
    const normalizedReasoningEffort = this.normalizeReasoningEffort();
    const completion = await this.openai.chat.completions.create({
      messages: messages.map((msg) => {
        const role = msg.role as "system" | "user" | "assistant";
        return {
          role,
          content:
            typeof msg.content === "string"
              ? msg.content
              : JSON.stringify(msg.content),
        };
      }),
      model: this.model,
      ...(normalizedReasoningEffort && {
        reasoning_effort: normalizedReasoningEffort,
      }),
    });
    const response = completion.choices[0].message;
    return {
      content: response.content || "",
      role: response.role,
    };
  }
}
