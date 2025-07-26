# Troubleshooting: "Offline" Issue

## Step 1: Check What's Happening

1. **Open your deployed frontend in a browser**
2. **Open Developer Tools (F12)**
3. **Go to the Console tab**
4. **Look for these debug messages:**
   - `API_BASE_URL: http://localhost:8000` (This means environment variable is not set)
   - `Health check error:` (This shows the actual error)

## Step 2: Quick Diagnostic

### If you see `API_BASE_URL: http://localhost:8000`:
This means the environment variable `VITE_API_BASE_URL` is not set in Vercel.

**Fix:**
1. Go to your Vercel dashboard
2. Select your frontend project
3. Go to Settings → Environment Variables
4. Add: `VITE_API_BASE_URL` = `https://your-backend-url.vercel.app`
5. Redeploy

### If you see CORS errors:
The backend needs CORS configuration.

**Fix:**
The backend already has CORS configured, but if you're still getting CORS errors, check that your backend is actually deployed.

### If you see "Failed to fetch" or network errors:
The backend URL is incorrect or the backend is not deployed.

**Fix:**
1. Deploy your backend first
2. Test the backend URL directly in browser
3. Update the frontend with the correct backend URL

## Step 3: Deploy Backend First

```bash
# Navigate to backend directory
cd Backend

# Deploy to Vercel
vercel

# Note the URL (e.g., https://your-backend-name.vercel.app)
```

## Step 4: Test Backend

1. **Visit your backend URL directly** (e.g., `https://your-backend-name.vercel.app/health`)
2. **You should see:** `{"status":"healthy","message":"API is running successfully"}`

If this doesn't work, your backend deployment failed.

## Step 5: Update Frontend

### Option A: Environment Variable (Recommended)

1. **In Vercel dashboard:**
   - Go to your frontend project
   - Settings → Environment Variables
   - Add: `VITE_API_BASE_URL` = `https://your-backend-url.vercel.app`
   - Redeploy

### Option B: Hardcode URL (Quick Fix)

Edit `Frontend/src/services/api.ts`:
```typescript
const API_BASE_URL = 'https://your-backend-url.vercel.app';
```

Then redeploy frontend.

## Step 6: Verify Fix

1. **Visit your frontend**
2. **Open Developer Tools (F12)**
3. **Check Console for:**
   - `API_BASE_URL: https://your-backend-url.vercel.app`
   - `Health check successful:`
4. **The status indicator should show "Online"**

## Common Error Messages

### "Failed to fetch"
- Backend not deployed
- Wrong backend URL
- Network connectivity issues

### "CORS error"
- Backend CORS not configured (should be fixed)
- Wrong backend URL

### "404 Not Found"
- Backend endpoint doesn't exist
- Wrong backend URL

### "500 Internal Server Error"
- Backend deployment failed
- Missing environment variables in backend
- Python dependencies not installed

## Debug Commands

### Check if backend is accessible:
```bash
curl https://your-backend-url.vercel.app/health
```

### Check environment variables in Vercel:
- Go to Vercel dashboard
- Project settings
- Environment variables section

### Check deployment logs:
- Go to Vercel dashboard
- Deployments tab
- Click on latest deployment
- Check build logs for errors

## Still Not Working?

1. **Check if you have API keys set up** in your backend
2. **Verify all dependencies** are in `requirement.txt`
3. **Check if your backend has the required files** (documents, etc.)
4. **Try deploying to a different platform** (Railway, Render, etc.)

## Emergency Fix

If nothing else works, you can temporarily disable the health check:

Edit `Frontend/src/hooks/use-api.ts`:
```typescript
export const useHealthCheck = () => {
  return useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: () => apiService.healthCheck(),
    refetchInterval: 30000,
    retry: 3,
    retryDelay: 1000,
    enabled: false, // Disable health check temporarily
  });
};
```

This will show the app as "Online" even if the backend is down, but at least users can see the interface. 