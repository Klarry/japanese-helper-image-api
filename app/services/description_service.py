from app.schemas.description import DescriptionResponse
from app.services.gemini_service import generate_text


def _uncontrolled_prompt(meaning: str) -> str:
    return f'Explain the concept represented by "{meaning}".'


def _controlled_prompt(meaning: str) -> str:
    return (
        f'Explain the concept represented by "{meaning}".\n'
        "Return exactly one sentence of no more than 15 words.\n"
        f'Use the phrase "{meaning}".\n'
        "End the response with <END>."
    )


async def describe(meaning: str) -> DescriptionResponse:
    uncontrolled_text = await generate_text(_uncontrolled_prompt(meaning))
    controlled_text = await generate_text(_controlled_prompt(meaning))

    return DescriptionResponse(
        uncontrolled=uncontrolled_text,
        controlled=controlled_text,
    )
