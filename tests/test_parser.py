from pathlib import Path

from summerizer.utils.parser import parse_file, parse_html, parse_raw_dir, write_parsed_json


def test_parse_html_removes_boilerplate_and_extracts_main_content():
    html = """
    <html>
      <head>
        <title>Example Page</title>
        <link rel="canonical" href="https://example.com/page" />
        <style>.hidden { display: none; }</style>
        <script>console.log("ignore me")</script>
      </head>
      <body>
        <header>Site Header</header>
        <main>
          <h1>Main Title</h1>
          <h2>Section One</h2>
          <p>This is the important content.</p>
        </main>
        <footer>Site Footer</footer>
      </body>
    </html>
    """

    document = parse_html(html, source_path="example.html")

    assert document.title == "Example Page"
    assert document.url == "https://example.com/page"
    assert document.headings == ["Main Title", "Section One"]
    assert "This is the important content." in document.text
    assert "Site Header" not in document.text
    assert "Site Footer" not in document.text
    assert "ignore me" not in document.text


def test_parse_html_removes_unwanted_heading_and_text_phrases():
    html = """
    <html>
      <head><title>Content Page</title></head>
      <body>
        <main>
          <h2>Breadcrumb</h2>
          <h1>Real Title</h1>
          <p>Breadcrumb Home Students Real content starts here.</p>
          <p>Related Tags: Visa, Travel</p>
          <p>Related Content</p>
          <p>Tools</p>
          <p>Social Media</p>
          <p>Tweets by StudyinStates</p>
        </main>
      </body>
    </html>
    """

    document = parse_html(html, source_path="content_page.html")

    assert document.headings == ["Real Title"]
    assert "Breadcrumb Home Students" not in document.text
    assert "Related Tags:" not in document.text
    assert "Related Content" not in document.text
    assert "Tools" not in document.text
    assert "Social Media" not in document.text
    assert "Tweets by StudyinStates" not in document.text
    assert "Real content starts here." in document.text


def test_parse_html_uses_h1_when_title_tag_is_missing():
    html = """
    <html>
      <body>
        <main>
          <h1>Fallback Heading</h1>
          <p>Body text.</p>
        </main>
      </body>
    </html>
    """

    document = parse_html(html, source_path="fallback_page.html")

    assert document.title == "Fallback Heading"
    assert document.headings == ["Fallback Heading"]
    assert "Body text." in document.text


def test_parse_html_uses_filename_when_title_and_h1_are_missing():
    html = """
    <html>
      <body>
        <main>
          <p>Only paragraph text.</p>
        </main>
      </body>
    </html>
    """

    document = parse_html(html, source_path="no_title_here.html")

    assert document.title == "no_title_here"
    assert document.headings == []
    assert "Only paragraph text." in document.text


def test_parse_file_reads_html_from_disk(tmp_path: Path):
    html_path = tmp_path / "page.html"
    html_path.write_text(
        """
        <html>
          <head><title>Disk Page</title></head>
          <body>
            <main><p>Loaded from file.</p></main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    document = parse_file(html_path)

    assert document.source_path == str(html_path)
    assert document.title == "Disk Page"
    assert "Loaded from file." in document.text


def test_parse_raw_dir_parses_all_html_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "first.html").write_text(
        "<html><body><main><h1>First</h1><p>One</p></main></body></html>",
        encoding="utf-8",
    )
    (raw_dir / "second.html").write_text(
        "<html><body><main><h1>Second</h1><p>Two</p></main></body></html>",
        encoding="utf-8",
    )

    documents = parse_raw_dir(raw_dir)

    assert len(documents) == 2
    assert [document.title for document in documents] == ["First", "Second"]


def test_parse_raw_dir_deduplicates_by_canonical_url(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    duplicate_html = """
    <html>
      <head>
        <title>Duplicate</title>
        <link rel="canonical" href="https://example.com/same-page" />
      </head>
      <body>
        <main><p>Duplicate page content.</p></main>
      </body>
    </html>
    """

    (raw_dir / "first.html").write_text(duplicate_html, encoding="utf-8")
    (raw_dir / "second.html").write_text(duplicate_html, encoding="utf-8")

    documents = parse_raw_dir(raw_dir)

    assert len(documents) == 1
    assert documents[0].url == "https://example.com/same-page"


def test_write_parsed_json_saves_cleaned_documents(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_path = tmp_path / "parsed" / "parsed_docs.json"

    (raw_dir / "page.html").write_text(
        """
        <html>
          <head>
            <title>Saved Page</title>
            <link rel="canonical" href="https://example.com/saved" />
          </head>
          <body>
            <main>
              <h1>Saved Page</h1>
              <p>Useful text.</p>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    written_path = write_parsed_json(out_path, raw_dir)

    assert written_path == out_path
    assert out_path.exists()

    contents = out_path.read_text(encoding="utf-8")
    assert "Saved Page" in contents
    assert "https://example.com/saved" in contents
    assert "Useful text." in contents
