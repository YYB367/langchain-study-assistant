"""预检 Chroma 检索，或批量调用本地 /ask 供人工评估。"""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "evaluation.md"
RESULTS_PATH = PROJECT_ROOT / "evaluation_api_results.json"
ASK_URL = "http://127.0.0.1:8000/ask"


def main() -> None:
    """选取题目、请求本地接口，并在每题结束后保存结果。"""
    parser = argparse.ArgumentParser(description="批量保存 /ask 的答案，供人工检查")
    parser.add_argument("ids", nargs="*", type=int, help="要运行的题号；留空时运行前三道待测题")
    parser.add_argument("--all", action="store_true", help="运行所有尚未保存结果的待测题")
    parser.add_argument("--retrieval-only", action="store_true", help="只检查检索页码，不调用 DeepSeek")
    args = parser.parse_args()
    if args.all and args.ids:
        parser.error("--all 不能与具体题号同时使用")
    if args.retrieval_only and args.all:
        parser.error("--retrieval-only 不能与 --all 同时使用；不指定题号时会检查全部题目")

    questions = {}
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) >= 7 and cells[0].isdigit():
            questions[int(cells[0])] = {
                "question": cells[1],
                "expected_page": cells[3],
                "judgment": cells[6],
            }

    if args.retrieval_only:
        from rag_service import load_saved_vector_store, retrieve_documents

        vector_store = load_saved_vector_store()
        for number in args.ids or questions:
            if number not in questions:
                print(f"跳过第 {number} 题：题号不存在")
                continue
            item = questions[number]
            documents = retrieve_documents(vector_store, item["question"])
            pages = sorted({doc.metadata["page"] + 1 for doc in documents})
            actual_pages = ", ".join(map(str, pages)) if pages else "无"
            expected_page = item["expected_page"]
            matched = (not pages) if expected_page == "无" else int(expected_page) in pages
            result = "页码符合" if matched else "待复核"
            print(f"第 {number:02d} 题：预期 {expected_page}，检索 {actual_pages}，{result}")
        return

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    pending = [
        number
        for number, item in questions.items()
        if item["judgment"] == "待测" and str(number) not in results
    ]
    selected = args.ids or (pending if args.all else pending[:3])

    if not selected:
        print("没有待运行的题目")
        return

    for number in selected:
        if number not in questions:
            print(f"跳过第 {number} 题：题号不存在")
            continue
        if questions[number]["judgment"] != "待测" or str(number) in results:
            print(f"跳过第 {number} 题：已有结果，避免重复调用模型")
            continue

        question = questions[number]["question"]
        body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
        request = Request(
            ASK_URL,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                status = response.status
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            status = error.code
            try:
                payload = json.loads(error.read().decode("utf-8"))
            except json.JSONDecodeError:
                payload = {"detail": "接口返回了无法解析的错误内容"}
        except (URLError, TimeoutError):
            print("连接本地接口失败，请确认 Uvicorn 正在运行；已完成的结果仍已保存")
            break

        results[str(number)] = {"question": question, "http_status": status, **payload}
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"第 {number} 题：HTTP {status}，"
            f"来源 {len(payload.get('sources', []))} 段，"
            f"耗时 {payload.get('elapsed_ms', '未返回')} ms"
        )

    print(f"结果文件：{RESULTS_PATH}")


if __name__ == "__main__":
    main()
