GPT4O_INPUT_PRICE_PER_1M = 2.50
GPT4O_OUTPUT_PRICE_PER_1M = 10.00

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * GPT4O_INPUT_PRICE_PER_1M
    output_cost = (output_tokens / 1_000_000) * GPT4O_OUTPUT_PRICE_PER_1M
    return round(input_cost + output_cost, 6)