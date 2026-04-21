export type UserRole = 'hr' | 'internal'

export interface LoginResponse {
  access_token: string
  token_type: string
  username: string
  role: UserRole
}

export interface MeResponse {
  username: string
  role: UserRole
}