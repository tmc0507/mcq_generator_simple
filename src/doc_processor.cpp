#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

namespace fs = std::filesystem;

struct SourceBlock {
    int index{};
    std::string type;
    std::string text;
};

struct Chunk {
    int id{};
    std::vector<int> source_indices;
    std::vector<std::string> source_types;
    std::string text;
};

static std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static std::string run_capture(const std::string& command) {
    std::array<char, 8192> buffer{};
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) throw std::runtime_error("Không thể chạy lệnh: " + command);
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
        result += buffer.data();
    }
    const int code = pclose(pipe);
    if (code != 0) throw std::runtime_error("Lệnh thất bại (code "
        + std::to_string(WIFEXITED(code) ? WEXITSTATUS(code) : code) + "): " + command);
    return result;
}

static std::string trim(std::string s) {
    auto is_space = [](unsigned char c) { return std::isspace(c); };
    while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static std::string xml_decode(std::string s) {
    const std::vector<std::pair<std::string, std::string>> ents = {
        {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}
    };
    for (const auto& [a, b] : ents) {
        size_t pos = 0;
        while ((pos = s.find(a, pos)) != std::string::npos) {
            s.replace(pos, a.size(), b);
            pos += b.size();
        }
    }
    return s;
}

static std::string clean_text(std::string s) {
    // Loại soft hyphen UTF-8 C2 AD
    const std::string soft_hyphen = "\xC2\xAD";
    size_t pos = 0;
    while ((pos = s.find(soft_hyphen, pos)) != std::string::npos) s.erase(pos, soft_hyphen.size());

    // Ghép từ bị ngắt dòng dạng: N-\ndimensional -> N-dimensional
    s = std::regex_replace(s, std::regex(R"(-\s*\n\s*)"), "-");

    // Chuẩn hóa khoảng trắng nhưng giữ ngắt đoạn.
    std::istringstream iss(s);
    std::ostringstream oss;
    std::string line;
    bool previous_blank = false;
    while (std::getline(iss, line)) {
        line = std::regex_replace(line, std::regex(R"([\t ]+)"), " ");
        line = trim(line);
        if (line.empty()) {
            if (!previous_blank) oss << '\n';
            previous_blank = true;
        } else {
            oss << line << '\n';
            previous_blank = false;
        }
    }
    return trim(oss.str());
}

static std::vector<SourceBlock> read_pdf(const fs::path& path) {
    // pdftotext chèn form-feed (\f) giữa các trang.
    const std::string cmd = "pdftotext -layout " + shell_quote(path.string()) + " -";
    const std::string raw = run_capture(cmd);

    std::vector<SourceBlock> blocks;
    std::stringstream ss(raw);
    std::string page;
    int page_no = 1;
    while (std::getline(ss, page, '\f')) {
        page = clean_text(page);
        if (!page.empty()) blocks.push_back({page_no, "page", page});
        ++page_no;
    }
    return blocks;
}

static std::vector<std::string> list_zip_entries(const fs::path& path) {
    const std::string raw = run_capture("unzip -Z1 " + shell_quote(path.string()));
    std::vector<std::string> entries;
    std::istringstream iss(raw);
    std::string line;
    while (std::getline(iss, line)) {
        line = trim(line);
        if (!line.empty()) entries.push_back(line);
    }
    return entries;
}

static std::string unzip_entry(const fs::path& path, const std::string& entry) {
    return run_capture("unzip -p " + shell_quote(path.string()) + " " + shell_quote(entry));
}

static std::string extract_xml_text(const std::string& xml, const std::regex& tag_regex) {
    std::string out;
    auto begin = std::sregex_iterator(xml.begin(), xml.end(), tag_regex);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        std::string t = xml_decode((*it)[1].str());
        if (!t.empty()) {
            if (!out.empty()) out += ' ';
            out += t;
        }
    }
    return clean_text(out);
}

static std::vector<SourceBlock> read_pptx(const fs::path& path) {
    std::regex slide_re(R"(^ppt/slides/slide([0-9]+)\.xml$)");
    std::regex text_re(R"(<a:t(?:\s[^>]*)?>([\s\S]*?)</a:t>)");
    std::map<int, std::string> slide_entries;

    for (const auto& entry : list_zip_entries(path)) {
        std::smatch m;
        if (std::regex_match(entry, m, slide_re)) slide_entries[std::stoi(m[1].str())] = entry;
    }

    std::vector<SourceBlock> blocks;
    for (const auto& [index, entry] : slide_entries) {
        std::string text = extract_xml_text(unzip_entry(path, entry), text_re);
        if (!text.empty()) blocks.push_back({index, "slide", text});
    }
    return blocks;
}

static std::vector<SourceBlock> read_docx(const fs::path& path) {
    const std::string xml = unzip_entry(path, "word/document.xml");
    std::regex para_re(R"(<w:p(?:\s[^>]*)?>([\s\S]*?)</w:p>)");
    std::regex text_re(R"(<w:t(?:\s[^>]*)?>([\s\S]*?)</w:t>)");

    std::vector<SourceBlock> blocks;
    int index = 1;
    auto begin = std::sregex_iterator(xml.begin(), xml.end(), para_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        std::string text = extract_xml_text((*it)[1].str(), text_re);
        if (!text.empty()) blocks.push_back({index++, "section", text});
    }
    return blocks;
}

static std::vector<SourceBlock> read_csv(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Không thể mở CSV: " + path.string());
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool quoted = false;
    for (char c; file.get(c);) {
        if (quoted) {
            if (c == '"' && file.peek() == '"') { field += '"'; file.get(); }
            else if (c == '"') quoted = false;
            else field += c;
        } else if (c == '"' && field.empty()) quoted = true;
        else if (c == ',') { row.push_back(trim(field)); field.clear(); }
        else if (c == '\n') {
            row.push_back(trim(field)); field.clear();
            rows.push_back(std::move(row)); row.clear();
        } else if (c != '\r') field += c;
    }
    if (quoted) throw std::runtime_error("CSV có dấu nháy kép chưa đóng");
    if (!field.empty() || !row.empty()) { row.push_back(trim(field)); rows.push_back(std::move(row)); }
    if (rows.empty()) return {};
    if (rows[0][0].rfind("\xEF\xBB\xBF", 0) == 0) rows[0][0].erase(0, 3);
    const auto headers = rows[0];
    std::vector<SourceBlock> blocks;
    for (size_t i = 1; i < rows.size(); ++i) {
        std::string text;
        for (size_t j = 0; j < rows[i].size(); ++j) {
            if (rows[i][j].empty()) continue;
            if (!text.empty()) text += " | ";
            text += (j < headers.size() && !headers[j].empty() ? headers[j] : "column " + std::to_string(j + 1));
            text += ": " + rows[i][j];
        }
        if (!text.empty()) blocks.push_back({static_cast<int>(i + 1), "row", clean_text(text)});
    }
    return blocks;
}

static std::vector<SourceBlock> read_legacy_office(const fs::path& path, const std::string& format) {
    std::string name = (fs::temp_directory_path() / "mcq_office_XXXXXX").string();
    std::vector<char> buffer(name.begin(), name.end());
    buffer.push_back('\0');
    char* created = mkdtemp(buffer.data());
    if (!created) throw std::runtime_error("Không thể tạo thư mục tạm để chuyển đổi Office");
    const fs::path temporary(created);
    try {
        const std::string command = "libreoffice -env:UserInstallation="
            + shell_quote("file://" + (temporary / "profile").string())
            + " --headless --convert-to " + format + " --outdir "
            + shell_quote(temporary.string()) + " " + shell_quote(fs::absolute(path).string());
        try {
            run_capture(command);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("LibreOffice không chuyển được " + path.string()
                + "; hãy kiểm tra LibreOffice hoặc lưu lại thành ." + format + ". Chi tiết: " + e.what());
        }
        fs::path converted = temporary / path.filename();
        converted.replace_extension("." + format);
        if (!fs::exists(converted)) throw std::runtime_error("LibreOffice không tạo được " + converted.string()
            + "; hãy kiểm tra gói " + (format == "pptx" ? "libreoffice-impress" : "libreoffice-writer"));
        auto blocks = format == "docx" ? read_docx(converted) : read_pptx(converted);
        fs::remove_all(temporary);
        return blocks;
    } catch (...) {
        fs::remove_all(temporary);
        throw;
    }
}

static std::vector<SourceBlock> read_document(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (ext == ".pdf") return read_pdf(path);
    if (ext == ".pptx") return read_pptx(path);
    if (ext == ".docx") return read_docx(path);
    if (ext == ".csv") return read_csv(path);
    if (ext == ".ppt") return read_legacy_office(path, "pptx");
    if (ext == ".doc") return read_legacy_office(path, "docx");
    throw std::runtime_error("Định dạng chưa hỗ trợ: " + ext);
}

static std::string label_for(const SourceBlock& b) {
    std::string t = b.type;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return std::toupper(c); });
    return "=== " + t + " " + std::to_string(b.index) + " ===\n";
}

static std::vector<Chunk> make_chunks(const std::vector<SourceBlock>& blocks, size_t max_chars, size_t max_blocks) {
    std::vector<Chunk> chunks;
    Chunk cur;
    cur.id = 1;

    auto flush = [&]() {
        if (!cur.text.empty()) {
            chunks.push_back(cur);
            cur = Chunk{};
            cur.id = static_cast<int>(chunks.size()) + 1;
        }
    };

    for (const auto& b : blocks) {
        const std::string piece = label_for(b) + b.text + "\n\n";
        if (!cur.text.empty() && (cur.text.size() + piece.size() > max_chars || cur.source_indices.size() >= max_blocks)) flush();

        // Nếu một block riêng lẻ rất dài thì chia thô theo max_chars.
        if (piece.size() > max_chars) {
            flush();
            size_t start = 0;
            while (start < piece.size()) {
                Chunk c;
                c.id = static_cast<int>(chunks.size()) + 1;
                c.source_indices = {b.index};
                c.source_types = {b.type};
                c.text = piece.substr(start, std::min(max_chars, piece.size() - start));
                chunks.push_back(std::move(c));
                start += max_chars;
            }
            cur.id = static_cast<int>(chunks.size()) + 1;
            continue;
        }

        cur.source_indices.push_back(b.index);
        cur.source_types.push_back(b.type);
        cur.text += piece;
    }
    flush();
    return chunks;
}

static std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) {
                    const char* hex = "0123456789abcdef";
                    o << "\\u00" << hex[(c >> 4) & 0xF] << hex[c & 0xF];
                } else {
                    o << static_cast<char>(c);
                }
        }
    }
    return o.str();
}

static void write_json(const fs::path& out_path, const fs::path& source, const std::vector<SourceBlock>& blocks,
                       const std::vector<Chunk>& chunks) {
    fs::create_directories(out_path.parent_path().empty() ? fs::path(".") : out_path.parent_path());
    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("Không thể ghi file: " + out_path.string());

    out << "{\n";
    out << "  \"document\": \"" << json_escape(source.filename().string()) << "\",\n";
    out << "  \"blocks\": [\n";
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto& b = blocks[i];
        out << "    {\"index\":" << b.index << ",\"type\":\"" << json_escape(b.type)
            << "\",\"text\":\"" << json_escape(b.text) << "\"}";
        if (i + 1 < blocks.size()) out << ',';
        out << '\n';
    }
    out << "  ],\n";
    out << "  \"chunks\": [\n";
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& c = chunks[i];
        out << "    {\"id\":" << c.id << ",\"source_indices\":[";
        for (size_t j = 0; j < c.source_indices.size(); ++j) {
            if (j) out << ',';
            out << c.source_indices[j];
        }
        out << "],\"source_types\":[";
        for (size_t j = 0; j < c.source_types.size(); ++j) {
            if (j) out << ',';
            out << "\"" << json_escape(c.source_types[j]) << "\"";
        }
        out << "],\"text\":\"" << json_escape(c.text) << "\"}";
        if (i + 1 < chunks.size()) out << ',';
        out << '\n';
    }
    out << "  ]\n}\n";
}

static void print_usage() {
    std::cout << "Usage:\n"
              << "  doc_processor FILE.(pdf|pptx|docx|csv|ppt|doc) --output work/chunks.json [--blocks-per-chunk 3] [--chunk-chars 12000]\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage();
            return 1;
        }

        fs::path input = argv[1];
        fs::path output = "work/chunks.json";
        size_t chunk_chars = 12000;
        size_t blocks_per_chunk = 3;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) output = argv[++i];
            else if (arg == "--chunk-chars" && i + 1 < argc) chunk_chars = static_cast<size_t>(std::stoul(argv[++i]));
            else if (arg == "--blocks-per-chunk" && i + 1 < argc) blocks_per_chunk = static_cast<size_t>(std::stoul(argv[++i]));
            else if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
            else throw std::runtime_error("Tham số không hợp lệ: " + arg);
        }

        if (!fs::exists(input)) throw std::runtime_error("Không tìm thấy file: " + input.string());
        if (chunk_chars < 1000) throw std::runtime_error("--chunk-chars quá nhỏ; dùng >= 1000.");
        if (blocks_per_chunk < 1) throw std::runtime_error("--blocks-per-chunk phải >= 1.");

        std::cout << "Reading: " << input << '\n';
        auto blocks = read_document(input);
        if (blocks.empty()) {
            throw std::runtime_error("Không extract được text. Nếu PDF/slide chỉ là ảnh thì V1 chưa có OCR.");
        }

        auto chunks = make_chunks(blocks, chunk_chars, blocks_per_chunk);
        write_json(output, input, blocks, chunks);

        std::cout << "Blocks : " << blocks.size() << '\n';
        std::cout << "Chunks : " << chunks.size() << '\n';
        std::cout << "Saved  : " << output << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 2;
    }
}
