from rag_legal_advisor import retrieve_legal_context, format_legal_context

snippets = retrieve_legal_context("厂房里违规电焊作业，引燃聚氨酯泡沫造成火灾", top_k=5)
print(f"检索到 {len(snippets)} 条")
print(format_legal_context(snippets)[:500])