// typescript_demo.ts
export type SessionId = string;

export interface UserSession {
    id: SessionId;
    userId: string;
    createdAt: Date;
}

export class SessionManager {
    private sessions = new Map<SessionId, UserSession>();

    public createSession(userId: string): UserSession {
        const id: SessionId = Math.random().toString(36).substring(2);
        const session: UserSession = { id, userId, createdAt: new Date() };
        this.sessions.set(id, session);
        return session;
    }
}

export function validateToken(token: string): boolean {
    return token.length > 16;
}
