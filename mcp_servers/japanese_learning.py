"""A small local MCP server with Japanese-learning tools, spoken to over stdio.

Day 16 needs something on the other end of an MCP connection to prove the
client works: that it can start a server, shake hands with it and read back
what it offers. This is that server. It is a separate program, not part of
the FastAPI app - the client starts it as a subprocess and talks to it over
its stdin and stdout, exactly as it would to any third-party MCP server.

Everything it knows is in the dictionaries below. No network, no Gemini, no
files: a server that exists to be connected to should never fail for a
reason that has nothing to do with the connection.

Run it on its own with ``python mcp_servers/japanese_learning.py`` - it will
sit waiting for a client on stdin. Nothing may be printed to stdout here: in
the stdio transport stdout *is* the protocol, and a stray print would corrupt
it.
"""

from pydantic import BaseModel

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SERVER_NAME = "japanese-learning"
SERVER_VERSION = "1.0.0"

server = MCPServer(
    SERVER_NAME,
    version=SERVER_VERSION,
    instructions="Look up Japanese words and kanji, and build example sentences for a learner.",
)


class WordEntry(BaseModel):
    word: str
    reading: str
    meaning_ru: str
    meaning_en: str
    jlpt_level: str


class KanjiInfo(BaseModel):
    kanji: str
    meaning_ru: str
    meaning_en: str
    onyomi: list[str]
    kunyomi: list[str]
    strokes: int
    jlpt_level: str


class ExampleSentence(BaseModel):
    word: str
    japanese: str
    reading: str
    translation_ru: str


_WORDS = [
    WordEntry(word="学生", reading="がくせい", meaning_ru="студент", meaning_en="student", jlpt_level="N5"),
    WordEntry(word="学校", reading="がっこう", meaning_ru="школа", meaning_en="school", jlpt_level="N5"),
    WordEntry(word="先生", reading="せんせい", meaning_ru="учитель", meaning_en="teacher", jlpt_level="N5"),
    WordEntry(word="勉強", reading="べんきょう", meaning_ru="учёба", meaning_en="study", jlpt_level="N5"),
    WordEntry(word="学習", reading="がくしゅう", meaning_ru="изучение", meaning_en="learning", jlpt_level="N3"),
    WordEntry(word="日本語", reading="にほんご", meaning_ru="японский язык", meaning_en="Japanese language", jlpt_level="N5"),
    WordEntry(word="水", reading="みず", meaning_ru="вода", meaning_en="water", jlpt_level="N5"),
    WordEntry(word="山", reading="やま", meaning_ru="гора", meaning_en="mountain", jlpt_level="N5"),
]

_KANJI = {
    "学": KanjiInfo(kanji="学", meaning_ru="учёба", meaning_en="study, learning", onyomi=["ガク"],
                   kunyomi=["まな.ぶ"], strokes=8, jlpt_level="N5"),
    "生": KanjiInfo(kanji="生", meaning_ru="жизнь, рождение", meaning_en="life, birth", onyomi=["セイ", "ショウ"],
                   kunyomi=["い.きる", "う.まれる", "なま"], strokes=5, jlpt_level="N5"),
    "日": KanjiInfo(kanji="日", meaning_ru="солнце, день", meaning_en="sun, day", onyomi=["ニチ", "ジツ"],
                   kunyomi=["ひ", "か"], strokes=4, jlpt_level="N5"),
    "水": KanjiInfo(kanji="水", meaning_ru="вода", meaning_en="water", onyomi=["スイ"],
                   kunyomi=["みず"], strokes=4, jlpt_level="N5"),
    "山": KanjiInfo(kanji="山", meaning_ru="гора", meaning_en="mountain", onyomi=["サン"],
                   kunyomi=["やま"], strokes=3, jlpt_level="N5"),
    "語": KanjiInfo(kanji="語", meaning_ru="слово, язык", meaning_en="word, language", onyomi=["ゴ"],
                   kunyomi=["かた.る"], strokes=14, jlpt_level="N5"),
}

_SENTENCES = {
    "学生": ExampleSentence(word="学生", japanese="私は学生です。", reading="わたしはがくせいです。",
                          translation_ru="Я студент."),
    "学校": ExampleSentence(word="学校", japanese="毎日学校に行きます。", reading="まいにちがっこうにいきます。",
                          translation_ru="Я каждый день хожу в школу."),
    "先生": ExampleSentence(word="先生", japanese="田中先生はやさしいです。", reading="たなかせんせいはやさしいです。",
                          translation_ru="Учитель Танака добрый."),
    "勉強": ExampleSentence(word="勉強", japanese="図書館で勉強します。", reading="としょかんでべんきょうします。",
                          translation_ru="Я занимаюсь в библиотеке."),
    "学習": ExampleSentence(word="学習", japanese="日本語の学習はとても楽しいです。",
                          reading="にほんごのがくしゅうはとてもたのしいです。",
                          translation_ru="Изучать японский очень интересно."),
    "水": ExampleSentence(word="水", japanese="水を一杯ください。", reading="みずをいっぱいください。",
                         translation_ru="Стакан воды, пожалуйста."),
}


@server.tool(title="Search Japanese word")
def search_japanese_word(query: str) -> list[WordEntry]:
    """Find Japanese words in the learner's dictionary by kanji, kana reading,
    Russian or English meaning. Returns every entry that matches, with its
    reading, meanings and JLPT level; an empty list means nothing matched."""
    needle = query.strip().lower()

    if not needle:
        return []

    return [
        entry
        for entry in _WORDS
        if needle in entry.word
        or needle in entry.reading
        or needle in entry.meaning_ru.lower()
        or needle in entry.meaning_en.lower()
    ]


@server.tool(title="Get kanji info")
def get_kanji_info(kanji: str) -> KanjiInfo:
    """Describe a single kanji: its meaning, on'yomi and kun'yomi readings,
    stroke count and JLPT level."""
    character = kanji.strip()

    if character not in _KANJI:
        # ToolError, not ValueError: an anticipated failure, so its text reaches
        # the caller. Any other exception is treated as a crash and withheld.
        raise ToolError(f"Kanji {character!r} is not in the local dictionary")

    return _KANJI[character]


@server.tool(title="Create example sentence")
def create_example_sentence(word: str) -> ExampleSentence:
    """Give a short example sentence that uses the word, with its kana reading
    and a Russian translation, so the learner sees the word in context."""
    key = word.strip()

    if key in _SENTENCES:
        return _SENTENCES[key]

    return ExampleSentence(
        word=key,
        japanese=f"「{key}」という言葉を勉強しています。",
        reading=f"「{key}」ということばをべんきょうしています。",
        translation_ru=f"Я изучаю слово «{key}».",
    )


if __name__ == "__main__":
    server.run("stdio")
