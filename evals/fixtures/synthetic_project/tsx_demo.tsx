// tsx_demo.tsx
import React, { useState, useEffect } from 'react';

export interface UserProfileProps {
    username: string;
    role: string;
}

export function UserProfileCard({ username, role }: UserProfileProps) {
    return (
        <div className="profile-card">
            <h2>{username}</h2>
            <p>Role: {role}</p>
        </div>
    );
}

export function useAuthStatus(token: string) {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    useEffect(() => {
        setIsLoggedIn(token.length > 0);
    }, [token]);
    return isLoggedIn;
}
