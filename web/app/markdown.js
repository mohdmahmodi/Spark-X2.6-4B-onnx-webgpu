// Markdown to sanitized HTML for model replies.
import { marked } from "marked";
import DOMPurify from "dompurify";

export const renderMarkdown = (text) => DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: false }));
