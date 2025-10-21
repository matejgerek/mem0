import OpenAI from "openai";
import { Embedder } from "./base";
import { EmbeddingConfig } from "../types";

export class OpenAIEmbedder implements Embedder {
  private openai: OpenAI;
  private model: string;
  private embeddingDims?: number;

  constructor(config: EmbeddingConfig) {
    const defaultHeaders =
      (config as any).headers ||
      (config.modelProperties?.headers as Record<string, string> | undefined);
    this.openai = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
      ...(defaultHeaders && { defaultHeaders }),
    });
    this.model = config.model || "text-embedding-3-small";
    this.embeddingDims = config.embeddingDims || 1536;
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.openai.embeddings.create({
      model: this.model,
      input: text,
    });
    return response.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await this.openai.embeddings.create({
      model: this.model,
      input: texts,
    });
    return response.data.map((item) => item.embedding);
  }
}
