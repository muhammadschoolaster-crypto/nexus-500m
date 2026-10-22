-- NEXUS demo schema
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    approved INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO conversations (title) VALUES ('First offline chat');

SELECT c.title, COUNT(m.id) AS memory_count
FROM conversations c
LEFT JOIN memories m ON m.id % 2 = 0
GROUP BY c.title
HAVING memory_count >= 0
ORDER BY c.title DESC;
