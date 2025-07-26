# Anushilon Bot - AI Assistant

A modern, bilingual (Bengali/English) AI chat assistant built with React, TypeScript, and Tailwind CSS. Features a responsive design with a collapsible sidebar for chat history management.

## 🌟 Features

- **Bilingual Support**: Full Bengali and English language support
- **Modern UI**: Clean, responsive design with dark/light theme toggle
- **Chat History**: Persistent chat sessions with sidebar management
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Real-time Chat**: Interactive chat interface with loading states
- **Accessibility**: Screen reader friendly with proper ARIA labels

## 🚀 Tech Stack

- **Frontend**: React 18 + TypeScript
- **Styling**: Tailwind CSS + shadcn/ui components
- **Build Tool**: Vite
- **State Management**: React Query (TanStack Query)
- **Routing**: React Router DOM
- **Icons**: Lucide React
- **Package Manager**: npm/bun

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd anushilon-bot
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   bun install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   # or
   bun dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:5173`

## 🎯 Usage

### Starting a New Chat
- Click the "নতুন চ্যাট | New Chat" button in the sidebar
- Or use the existing chat session

### Sending Messages
- Type your question in Bengali or English
- Press Enter or click the send button
- The AI will respond in the same language

### Managing Chat History
- **View History**: All chat sessions are listed in the sidebar
- **Switch Chats**: Click on any chat in the sidebar to switch
- **Delete Chats**: Use the three-dot menu next to each chat
- **Toggle Sidebar**: Press 'B' key or click the menu button

### Theme Toggle
- Click the moon/sun icon in the top-right corner
- Supports light and dark themes

## 🏗️ Project Structure

```
anushilon-bot/
├── src/
│   ├── components/
│   │   ├── ui/                 # shadcn/ui components
│   │   ├── ChatInterface.tsx   # Main chat interface
│   │   ├── ChatSidebar.tsx     # Sidebar component
│   │   ├── ThemeToggle.tsx     # Theme switcher
│   │   └── ...
│   ├── pages/
│   │   ├── Index.tsx           # Main page
│   │   └── NotFound.tsx        # 404 page
│   ├── hooks/
│   │   ├── use-mobile.tsx      # Mobile detection
│   │   └── use-toast.ts        # Toast notifications
│   ├── lib/
│   │   └── utils.ts            # Utility functions
│   └── main.tsx                # App entry point
├── public/                     # Static assets
├── index.html                  # HTML template
└── package.json                # Dependencies
```

## 🎨 Customization

### Adding API Integration
To connect to your AI backend, modify the `handleSendMessage` function in `ChatInterfaceWithSidebar.tsx`:

```typescript
// Replace the simulated response with your API call
const response = await fetch('YOUR_API_ENDPOINT', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: inputMessage })
});
const data = await response.json();
```

### Styling
- **Colors**: Modify CSS variables in `src/index.css`
- **Components**: Customize shadcn/ui components in `src/components/ui/`
- **Layout**: Adjust responsive breakpoints in Tailwind config

### Adding New Languages
1. Update the welcome message in `ChatInterfaceWithSidebar.tsx`
2. Add language-specific placeholders
3. Update the tip message for keyboard shortcuts

## 🔧 Development

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
```

### Code Style
- TypeScript for type safety
- ESLint for code quality
- Prettier for formatting
- Tailwind CSS for styling

## 🌐 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## 📱 Mobile Support

- Responsive design for all screen sizes
- Touch-friendly interface
- Mobile-optimized sidebar behavior
- Floating sidebar toggle for mobile devices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [shadcn/ui](https://ui.shadcn.com/) for beautiful UI components
- [Lucide](https://lucide.dev/) for icons
- [Tailwind CSS](https://tailwindcss.com/) for styling
- [Vite](https://vitejs.dev/) for fast development

